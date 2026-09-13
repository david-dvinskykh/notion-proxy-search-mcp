// Package daemon runs the long-lived half of the proxy: it owns the Notion
// token, the mirror, the embedding model and the sync loops, and answers the
// stdio processes over a unix socket. One daemon per board keeps the model in
// memory exactly once, which on an 8 GB Raspberry Pi shared with other services is the
// difference between 120 MB and 120 MB per session.
package daemon

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/embed"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/index"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/ipc"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mirror"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/notion"
	syncpkg "github.com/david-dvinskykh/notion-proxy-search-mcp/internal/sync"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/tools"
)

// Config holds everything the daemon reads from the environment.
type Config struct {
	Token      string
	DataDir    string
	ModelPath  string
	SocketPath string

	PollInterval      time.Duration
	ReconcileInterval time.Duration
	// EmbedBatch and EmbedPause throttle the indexing loop so it never starves
	// the other containers on the board.
	EmbedBatch int
	EmbedPause time.Duration
	// Workers caps the goroutines the embedding kernels use.
	Workers int
	// SkipEmbedSources lists data sources left out of the vector index. They
	// stay mirrored, queryable by SQL and searchable by keyword.
	SkipEmbedSources  []string
	RequestsPerSecond float64
	APIVersion        string
	// BaseURL overrides the Notion API root, for tests and for routing through
	// a proxy.
	BaseURL string
	Logger  *log.Logger
}

// MirrorPath is where the SQLite mirror lives.
func (c Config) MirrorPath() string { return filepath.Join(c.DataDir, "mirror.db") }

// Daemon is the running service.
type Daemon struct {
	cfg    Config
	db     *mirror.DB
	ro     *sql.DB
	idx    *index.Store
	syncer *syncpkg.Syncer
	svc    *tools.Service
	model  *embed.Model
	logger *log.Logger
}

// Open prepares the daemon: mirror, model, index cache, Notion client.
func Open(ctx context.Context, cfg Config) (*Daemon, error) {
	if cfg.Token == "" {
		return nil, errors.New("daemon: NOTION_TOKEN is empty; the daemon needs an internal integration secret with read access")
	}
	if err := os.MkdirAll(cfg.DataDir, 0o755); err != nil {
		return nil, err
	}
	logger := cfg.Logger
	if logger == nil {
		logger = log.New(os.Stderr, "", log.LstdFlags)
	}

	db, err := mirror.Open(ctx, cfg.MirrorPath())
	if err != nil {
		return nil, err
	}
	if err := db.EnsureViews(ctx); err != nil {
		db.Close()
		return nil, err
	}
	ro, err := openReadOnly(ctx, cfg.MirrorPath())
	if err != nil {
		db.Close()
		return nil, err
	}

	d := &Daemon{cfg: cfg, db: db, ro: ro, logger: logger}

	var emb index.Embedder
	if cfg.ModelPath != "" {
		if _, statErr := os.Stat(cfg.ModelPath); statErr == nil {
			if cfg.Workers > 0 {
				embed.SetWorkers(cfg.Workers)
			}
			model, loadErr := embed.Load(cfg.ModelPath)
			if loadErr != nil {
				d.Close()
				return nil, fmt.Errorf("daemon: load model %s: %w", cfg.ModelPath, loadErr)
			}
			d.model, emb = model, model
			logger.Printf("model %s loaded: %d dims, %d tokens max", model.Name(), model.Dim(), model.MaxTokens())
		} else {
			logger.Printf("no model at %s: running keyword-only until one is installed", cfg.ModelPath)
		}
	}

	d.idx = index.New(db, emb)
	if len(cfg.SkipEmbedSources) > 0 {
		d.idx.SkipSources(cfg.SkipEmbedSources)
		logger.Printf("vector index excludes %d data source(s): %s",
			len(cfg.SkipEmbedSources), strings.Join(cfg.SkipEmbedSources, ", "))
	}
	if err := d.idx.LoadVectors(ctx); err != nil {
		d.Close()
		return nil, err
	}
	logger.Printf("mirror %s opened: %d vectors cached", cfg.MirrorPath(), d.idx.VectorCount())

	client, err := notion.New(notion.Options{
		Token:             cfg.Token,
		BaseURL:           cfg.BaseURL,
		Version:           cfg.APIVersion,
		RequestsPerSecond: cfg.RequestsPerSecond,
		Logf:              logger.Printf,
	})
	if err != nil {
		d.Close()
		return nil, err
	}
	d.syncer = syncpkg.New(client, db, d.idx, syncpkg.Options{Logf: logger.Printf})
	d.svc = &tools.Service{DB: db, RO: ro, Index: d.idx, Syncer: d.syncer, Logf: logger.Printf}
	return d, nil
}

// Service exposes the operation set, so a single-process mode can serve MCP
// directly from the daemon without the socket.
func (d *Daemon) Service() *tools.Service { return d.svc }

// Close releases the mirror and the model.
func (d *Daemon) Close() error {
	if d.ro != nil {
		_ = d.ro.Close()
	}
	if d.model != nil {
		_ = d.model.Close()
	}
	if d.db != nil {
		return d.db.Close()
	}
	return nil
}

// Run starts the socket and the sync loops and blocks until ctx is done.
func (d *Daemon) Run(ctx context.Context) error {
	server, err := ipc.Listen(d.cfg.SocketPath, d, d.logger.Printf)
	if err != nil {
		return err
	}
	defer server.Close()
	d.logger.Printf("listening on %s", server.Addr())

	errs := make(chan error, 1)
	go func() { errs <- server.Serve(ctx) }()
	go d.syncLoop(ctx)
	go d.indexLoop(ctx)

	select {
	case <-ctx.Done():
		return nil
	case err := <-errs:
		return err
	}
}

// syncLoop bootstraps if needed and then polls.
func (d *Daemon) syncLoop(ctx context.Context) {
	bootstrapped, _ := d.db.Meta(ctx, "bootstrapped")
	if bootstrapped == "" {
		d.logger.Printf("bootstrap: first run, crawling the workspace")
		if err := d.bootstrap(ctx); err != nil {
			d.logger.Printf("bootstrap failed: %v", err)
		}
	}

	poll := time.NewTicker(d.cfg.PollInterval)
	defer poll.Stop()
	full := time.NewTicker(d.cfg.ReconcileInterval)
	defer full.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-poll.C:
			d.runPass(ctx, false)
		case <-full.C:
			d.runPass(ctx, true)
		}
	}
}

func (d *Daemon) bootstrap(ctx context.Context) error {
	if _, err := d.syncer.Discover(ctx); err != nil {
		return err
	}
	if err := d.syncer.SyncUsers(ctx); err != nil {
		d.logger.Printf("bootstrap: users: %v", err)
	}
	st, err := d.syncer.Pass(ctx, true)
	if err != nil {
		return err
	}
	d.db.Logf(ctx, nowISO(), "bootstrap", "%s", st.Describe())
	return d.db.SetMeta(ctx, "bootstrapped", nowISO())
}

func (d *Daemon) runPass(ctx context.Context, full bool) {
	if full {
		// Schemas change without any row changing; a new property has to reach
		// the collection:// view before a query can use it.
		if _, err := d.syncer.Discover(ctx); err != nil {
			d.logger.Printf("discover: %v", err)
		}
	}
	st, err := d.syncer.Pass(ctx, full)
	if err != nil {
		d.logger.Printf("sync pass: %v", err)
		d.db.Logf(ctx, nowISO(), "error", "sync pass: %v", err)
		return
	}
	kind := "pass"
	key := "last_pass"
	if full {
		kind, key = "full", "last_full_pass"
	}
	if st.Changed > 0 || st.Content > 0 || st.Trashed > 0 || full {
		d.db.Logf(ctx, nowISO(), kind, "%s", st.Describe())
	}
	_ = d.db.SetMeta(ctx, key, nowISO())
	if st.Trashed > 0 {
		if err := d.idx.RefreshMetadata(ctx); err != nil {
			d.logger.Printf("refresh metadata: %v", err)
		}
	}
}

// indexLoop keeps chunks and embeddings behind the mirror. It runs small
// batches with a pause between them: embedding is the only CPU-hungry thing
// here, and the board has other work.
func (d *Daemon) indexLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(d.cfg.EmbedPause):
		}

		stale, err := d.idx.StalePages(ctx, 20)
		if err != nil {
			d.logger.Printf("index: stale pages: %v", err)
			continue
		}
		for _, id := range stale {
			if err := d.idx.Reindex(ctx, id); err != nil {
				d.logger.Printf("index: reindex %s: %v", id, err)
			}
		}

		done, err := d.idx.EmbedPending(ctx, d.cfg.EmbedBatch)
		if err != nil && ctx.Err() == nil {
			d.logger.Printf("index: embed: %v", err)
		}
		if done > 0 {
			pending, _ := d.idx.PendingCount(ctx)
			d.logger.Printf("index: embedded %d chunks, %d pending", done, pending)
		}
	}
}

// Search implements ipc.Handler.
func (d *Daemon) Search(ctx context.Context, q index.Query) ([]index.Result, error) {
	return d.idx.Search(ctx, q)
}

// Resync implements ipc.Handler.
func (d *Daemon) Resync(ctx context.Context, pageID string, full bool) (string, error) {
	return d.svc.Resync(ctx, tools.ResyncArgs{PageID: pageID, Full: full})
}

// Status implements ipc.Handler.
func (d *Daemon) Status(ctx context.Context) (map[string]any, error) {
	return d.svc.Status(ctx)
}

// openReadOnly opens a second handle that cannot write, used for the SQL tool.
func openReadOnly(ctx context.Context, path string) (*sql.DB, error) {
	db, err := sql.Open("sqlite",
		"file:"+path+"?mode=ro&_pragma=busy_timeout(5000)&_pragma=query_only(1)&_time_format=sqlite")
	if err != nil {
		return nil, err
	}
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, err
	}
	return db, nil
}

func nowISO() string { return time.Now().UTC().Format(time.RFC3339) }
