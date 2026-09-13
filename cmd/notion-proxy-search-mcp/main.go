// Command notion-proxy-search-mcp is a read-only Notion proxy with local
// hybrid search, exposed over MCP on stdio.
//
// Subcommands:
//
//	serve    (default) the MCP stdio server the client spawns
//	daemon   the long-lived mirror: sync loops, embedding model, unix socket
//	convert  quantize a Hugging Face embedding checkpoint into a weight blob
//	status   print mirror status and exit
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/daemon"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/embed"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/index"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/ipc"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mcp"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mirror"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/tools"
)

// Version is overridden at build time with -ldflags.
var Version = "dev"

func main() {
	command := "serve"
	args := os.Args[1:]
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		command, args = args[0], args[1:]
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	var err error
	switch command {
	case "serve":
		err = runServe(ctx, args)
	case "daemon":
		err = runDaemon(ctx, args)
	case "convert":
		err = runConvert(args)
	case "status":
		err = runStatus(ctx, args)
	case "version":
		fmt.Println(Version)
	case "help", "-h", "--help":
		usage()
	default:
		usage()
		err = fmt.Errorf("unknown command %q", command)
	}
	if err != nil {
		// stdout belongs to the MCP transport; diagnostics go to stderr.
		fmt.Fprintf(os.Stderr, "notion-proxy-search-mcp: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `notion-proxy-search-mcp — read-only Notion mirror with local hybrid search

  serve      MCP server on stdio (default)
  daemon     sync loops, embedding model and unix socket
  convert    quantize a Hugging Face checkpoint into a weight blob
  status     print mirror status

Environment:
  NOTION_TOKEN             internal integration secret (daemon only, required)
  NPS_DATA_DIR             mirror and socket location (default /var/lib/notion-proxy-search-mcp)
  NPS_MODEL                weight blob path (default <data dir>/model.npse)
  NPS_POLL_INTERVAL        incremental sync period (default 30s)
  NPS_RECONCILE_INTERVAL   full pass period, detects deletions (default 6h)
  NPS_EMBED_BATCH          chunks embedded per indexing tick (default 8)
  NPS_EMBED_PAUSE          pause between indexing ticks (default 3s)
  NPS_EMBED_WORKERS        goroutines for the embedding kernels (default: all cores)
  NPS_EMBED_SKIP_SOURCES   data source ids left out of the vector index, comma
                           separated; they stay searchable by keyword and SQL
  NPS_RPS                  Notion requests per second (default 2.5)
  NPS_NOTION_VERSION       Notion-Version header (default 2025-09-03)
  NPS_NOTION_BASE          Notion API root override (default https://api.notion.com/v1)
  NPS_NO_DAEMON=1          do not auto-start the daemon from serve
`)
}

// config reads the environment shared by every subcommand.
func config() daemon.Config {
	dataDir := envOr("NPS_DATA_DIR", "/var/lib/notion-proxy-search-mcp")
	cfg := daemon.Config{
		Token:             os.Getenv("NOTION_TOKEN"),
		DataDir:           dataDir,
		ModelPath:         envOr("NPS_MODEL", filepath.Join(dataDir, "model.npse")),
		SocketPath:        envOr("NPS_SOCKET", filepath.Join(dataDir, "daemon.sock")),
		PollInterval:      envDuration("NPS_POLL_INTERVAL", 30*time.Second),
		ReconcileInterval: envDuration("NPS_RECONCILE_INTERVAL", 6*time.Hour),
		EmbedBatch:        envInt("NPS_EMBED_BATCH", 8),
		EmbedPause:        envDuration("NPS_EMBED_PAUSE", 3*time.Second),
		Workers:           envInt("NPS_EMBED_WORKERS", 0),
		SkipEmbedSources:  envList("NPS_EMBED_SKIP_SOURCES"),
		RequestsPerSecond: envFloat("NPS_RPS", 2.5),
		APIVersion:        envOr("NPS_NOTION_VERSION", ""),
		BaseURL:           envOr("NPS_NOTION_BASE", ""),
	}
	return cfg
}

// runServe is the MCP stdio server. It reads the mirror directly and asks the
// daemon only for the operations that need the model or the Notion token.
func runServe(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	noDaemon := fs.Bool("no-daemon", os.Getenv("NPS_NO_DAEMON") == "1", "do not auto-start the sync daemon")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg := config()
	logger := log.New(os.Stderr, "[nps serve] ", log.LstdFlags)

	client := ipc.NewClient(cfg.SocketPath)
	if !client.Available() && !*noDaemon {
		if err := startDaemon(cfg, logger); err != nil {
			logger.Printf("could not start the sync daemon: %v", err)
		}
	}

	if err := ensureMirror(ctx, cfg); err != nil {
		return err
	}
	db, err := mirror.OpenReadOnly(ctx, cfg.MirrorPath())
	if err != nil {
		return fmt.Errorf("open mirror: %w", err)
	}
	defer db.Close()

	// No embedder here on purpose: the weights are loaded once, in the daemon.
	svc := &tools.Service{
		DB:     db,
		RO:     db.SQL(),
		Index:  index.New(db, nil),
		Remote: client,
		Logf:   logger.Printf,
	}
	srv := mcp.NewServer("notion-proxy-search", Version, logger)
	svc.Register(srv)
	return srv.Serve(ctx, os.Stdin, os.Stdout)
}

// ensureMirror creates an empty mirror if the daemon has not made one yet, so
// the tools answer with empty results and a status instead of failing to start.
func ensureMirror(ctx context.Context, cfg daemon.Config) error {
	if _, err := os.Stat(cfg.MirrorPath()); err == nil {
		return nil
	}
	deadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(cfg.MirrorPath()); err == nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
	if err := os.MkdirAll(cfg.DataDir, 0o755); err != nil {
		return err
	}
	db, err := mirror.Open(ctx, cfg.MirrorPath())
	if err != nil {
		return fmt.Errorf("create mirror: %w", err)
	}
	return db.Close()
}

// startDaemon launches the daemon detached so it survives this session. A gateway
// spawns and kills the stdio process per session; the mirror must not go with it.
func startDaemon(cfg daemon.Config, logger *log.Logger) error {
	if cfg.Token == "" {
		return errors.New("NOTION_TOKEN is not set, so there is nothing to sync with")
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(cfg.DataDir, 0o755); err != nil {
		return err
	}
	logPath := filepath.Join(cfg.DataDir, "daemon.log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer logFile.Close()

	cmd := exec.Command(exe, "daemon")
	cmd.Env = os.Environ()
	cmd.Stdout, cmd.Stderr = logFile, logFile
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return err
	}
	// Do not wait for it: releasing the process makes it a child of init.
	go func() { _ = cmd.Process.Release() }()
	logger.Printf("started sync daemon (pid %d), log: %s", cmd.Process.Pid, logPath)

	client := ipc.NewClient(cfg.SocketPath)
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if client.Available() {
			return nil
		}
		time.Sleep(250 * time.Millisecond)
	}
	return fmt.Errorf("daemon did not open %s within 15s; see %s", cfg.SocketPath, logPath)
}

// runDaemon is the long-lived process.
func runDaemon(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("daemon", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	serveMCP := fs.Bool("stdio", false, "also serve MCP on stdio from this process (single-process mode)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg := config()
	cfg.Logger = log.New(os.Stderr, "[nps daemon] ", log.LstdFlags)

	d, err := daemon.Open(ctx, cfg)
	if err != nil {
		return err
	}
	defer d.Close()

	if *serveMCP {
		go func() {
			if err := d.Run(ctx); err != nil {
				cfg.Logger.Printf("daemon: %v", err)
			}
		}()
		srv := mcp.NewServer("notion-proxy-search", Version, cfg.Logger)
		d.Service().Register(srv)
		return srv.Serve(ctx, os.Stdin, os.Stdout)
	}
	return d.Run(ctx)
}

// runConvert turns a Hugging Face checkpoint into a quantized weight blob.
func runConvert(args []string) error {
	fs := flag.NewFlagSet("convert", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	modelDir := fs.String("model-dir", "", "directory with config.json, tokenizer.json and model.safetensors")
	out := fs.String("out", "model.npse", "output blob path")
	name := fs.String("name", "", "model name stored with every vector")
	dim := fs.Int("dim", 0, "truncate the pooled vector to this many dimensions (0 keeps all)")
	queryPrefix := fs.String("query-prefix", "query: ", "prefix added to search queries")
	passagePrefix := fs.String("passage-prefix", "passage: ", "prefix added to indexed text")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelDir == "" {
		fs.Usage()
		return errors.New("convert: -model-dir is required")
	}
	return embed.Convert(embed.ConvertOptions{
		ModelDir:      *modelDir,
		Out:           *out,
		ModelName:     *name,
		Dim:           *dim,
		QueryPrefix:   *queryPrefix,
		PassagePrefix: *passagePrefix,
		Logf:          func(format string, a ...any) { fmt.Fprintf(os.Stderr, format+"\n", a...) },
	})
}

// runStatus prints the mirror's state for a human.
func runStatus(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("status", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg := config()
	client := ipc.NewClient(cfg.SocketPath)
	if client.Available() {
		status, err := client.Status(ctx)
		if err != nil {
			return err
		}
		return printJSON(status)
	}
	if _, err := os.Stat(cfg.MirrorPath()); err != nil {
		return fmt.Errorf("no mirror at %s and no daemon on %s", cfg.MirrorPath(), cfg.SocketPath)
	}
	db, err := mirror.OpenReadOnly(ctx, cfg.MirrorPath())
	if err != nil {
		return err
	}
	defer db.Close()
	svc := &tools.Service{DB: db, RO: db.SQL(), Index: index.New(db, nil)}
	status, err := svc.Status(ctx)
	if err != nil {
		return err
	}
	return printJSON(status)
}

func printJSON(v any) error {
	encoded, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(encoded))
	return nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envDuration(key string, fallback time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		fmt.Fprintf(os.Stderr, "notion-proxy-search-mcp: %s=%q is not a duration, using %s\n", key, v, fallback)
		return fallback
	}
	return d
}

func envInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

// envList reads a comma-separated list, dropping blanks.
func envList(key string) []string {
	raw := os.Getenv(key)
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	var out []string
	for _, part := range strings.Split(raw, ",") {
		if part = strings.TrimSpace(part); part != "" {
			out = append(out, part)
		}
	}
	return out
}

func envFloat(key string, fallback float64) float64 {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return fallback
	}
	return f
}
