package daemon

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/index"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/ipc"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/tools"
)

// fakeNotion serves the smallest workspace that exercises the whole path: one
// data source, one row, one block.
func fakeNotion() http.Handler {
	mux := http.NewServeMux()
	list := func(w http.ResponseWriter, results string) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","results":[` + results + `],"has_more":false,"next_cursor":null}`))
	}
	mux.HandleFunc("/v1/search", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Filter struct {
				Value string `json:"value"`
			} `json:"filter"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		switch body.Filter.Value {
		case "data_source":
			list(w, `{"object":"data_source","id":"ds-facts",
				"title":[{"type":"text","plain_text":"Факты","annotations":{}}],
				"parent":{"type":"database_id","database_id":"db-facts"},
				"properties":{"Утверждение":{"type":"title"},"Доверие":{"type":"select"}}}`)
		default:
			list(w, "")
		}
	})
	mux.HandleFunc("/v1/databases/db-facts", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"object":"database","id":"db-facts",
			"title":[{"type":"text","plain_text":"Факты","annotations":{}}],
			"parent":{"type":"workspace","workspace":true},
			"data_sources":[{"id":"ds-facts","name":"Факты"}]}`))
	})
	mux.HandleFunc("/v1/data_sources/ds-facts/query", func(w http.ResponseWriter, r *http.Request) {
		list(w, `{"object":"page","id":"fact-1","created_time":"2026-09-01T00:00:00.000Z",
			"last_edited_time":"2026-09-04T14:47:00.000Z","archived":false,"in_trash":false,
			"url":"https://app.notion.com/p/fact-1",
			"parent":{"type":"data_source","data_source_id":"ds-facts"},
			"properties":{"Утверждение":{"type":"title","title":[{"type":"text","plain_text":"Контейнер gateway держит 3,1 ГБ ОЗУ","annotations":{}}]},
			              "Доверие":{"type":"select","select":{"name":"подтверждено"}}}}`)
	})
	mux.HandleFunc("/v1/blocks/", func(w http.ResponseWriter, r *http.Request) {
		list(w, `{"object":"block","id":"b1","type":"paragraph","has_children":false,
			"paragraph":{"rich_text":[{"type":"text","plain_text":"Замер 04.09.2026.","annotations":{}}]}}`)
	})
	mux.HandleFunc("/v1/users", func(w http.ResponseWriter, r *http.Request) {
		list(w, `{"object":"user","id":"u1","name":"Ada Lovelace","type":"person","person":{"email":"ada@example.com"}}`)
	})
	return mux
}

func newDaemon(t *testing.T) (*Daemon, Config) {
	t.Helper()
	srv := httptest.NewServer(fakeNotion())
	t.Cleanup(srv.Close)
	t.Setenv("NOTION_API_BASE", srv.URL+"/v1")

	dir := t.TempDir()
	cfg := Config{
		Token:             "secret_test",
		DataDir:           dir,
		ModelPath:         filepath.Join(dir, "absent.npse"),
		SocketPath:        filepath.Join(dir, "daemon.sock"),
		PollInterval:      50 * time.Millisecond,
		ReconcileInterval: time.Hour,
		EmbedBatch:        4,
		EmbedPause:        20 * time.Millisecond,
		RequestsPerSecond: 1000,
		BaseURL:           srv.URL + "/v1",
		Logger:            log.New(testWriter{t}, "[daemon] ", 0),
	}
	d, err := Open(context.Background(), cfg)
	if err != nil {
		t.Fatalf("open daemon: %v", err)
	}
	t.Cleanup(func() { d.Close() })
	return d, cfg
}

type testWriter struct{ t *testing.T }

func (w testWriter) Write(p []byte) (int, error) {
	w.t.Log(strings.TrimRight(string(p), "\n"))
	return len(p), nil
}

func TestDaemonBootstrapsAndServesOverSocket(t *testing.T) {
	d, cfg := newDaemon(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		if err := d.Run(ctx); err != nil {
			t.Errorf("run: %v", err)
		}
	}()

	client := ipc.NewClient(cfg.SocketPath)
	if !waitFor(2*time.Second, client.Available) {
		t.Fatal("the daemon never opened its socket")
	}

	// The bootstrap crawl runs on start; wait for the row to land.
	if !waitFor(5*time.Second, func() bool {
		results, err := client.Search(ctx, index.Query{Text: "gateway ОЗУ", Mode: index.ModeKeyword, Limit: 3})
		return err == nil && len(results) > 0
	}) {
		t.Fatal("the bootstrap crawl never produced a searchable row")
	}

	results, err := client.Search(ctx, index.Query{Text: "gateway ОЗУ", Mode: index.ModeKeyword, Limit: 3})
	if err != nil {
		t.Fatal(err)
	}
	if results[0].PageID != "fact-1" {
		t.Fatalf("results = %+v", results)
	}

	status, err := client.Status(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if status["served_by"] != "daemon" {
		t.Fatalf("status = %+v", status)
	}

	// SQL over the mirrored collection view must work from the daemon too.
	rows, err := d.Service().Query(ctx, queryArgsFor(`SELECT "Доверие" FROM "collection://ds-facts"`))
	if err != nil {
		t.Fatal(err)
	}
	if rows.RowCount != 1 || rows.Rows[0][0] != "подтверждено" {
		t.Fatalf("rows = %+v", rows)
	}
}

func TestDaemonRefusesWithoutToken(t *testing.T) {
	dir := t.TempDir()
	_, err := Open(context.Background(), Config{DataDir: dir, SocketPath: filepath.Join(dir, "s.sock")})
	if err == nil {
		t.Fatal("a daemon without a token must not start")
	}
	if !strings.Contains(err.Error(), "NOTION_TOKEN") {
		t.Fatalf("unhelpful error: %v", err)
	}
}

func TestDaemonRunsWithoutAModel(t *testing.T) {
	d, _ := newDaemon(t)
	status, err := d.Status(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got := status["embedding_model"]; got != "none (keyword search only)" {
		t.Fatalf("embedding_model = %v", got)
	}
}

func queryArgsFor(statement string) tools.QueryArgs {
	return tools.QueryArgs{Query: statement}
}

func waitFor(limit time.Duration, cond func() bool) bool {
	deadline := time.Now().Add(limit)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(25 * time.Millisecond)
	}
	return false
}
