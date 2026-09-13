package sync

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	stdsync "sync"
	"testing"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/index"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mirror"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/notion"
)

// fakeNotion is a minimal stand-in for the API: one database with one data
// source, two rows, and block content for one of them.
type fakeNotion struct {
	mu         stdsync.Mutex
	rows       map[string]json.RawMessage
	rowOrder   []string
	blocks     map[string][]json.RawMessage
	queries    int
	lastFilter map[string]any
}

func row(id, title, edited string, extra string) json.RawMessage {
	props := `"Утверждение":{"type":"title","title":[{"type":"text","plain_text":"` + title + `","annotations":{}}]}`
	if extra != "" {
		props += "," + extra
	}
	return json.RawMessage(fmt.Sprintf(`{
		"object":"page","id":%q,"created_time":"2026-09-01T00:00:00.000Z",
		"last_edited_time":%q,"archived":false,"in_trash":false,
		"url":"https://app.notion.com/p/%s",
		"parent":{"type":"data_source","data_source_id":"ds-facts"},
		"properties":{%s}}`, id, edited, id, props))
}

func (f *fakeNotion) handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/v1/search", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Filter struct {
				Value string `json:"value"`
			} `json:"filter"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		switch body.Filter.Value {
		case "data_source":
			writeList(w, []json.RawMessage{json.RawMessage(`{
				"object":"data_source","id":"ds-facts",
				"title":[{"type":"text","plain_text":"🧩 Заметки","annotations":{}}],
				"parent":{"type":"database_id","database_id":"db-facts"},
				"properties":{"Утверждение":{"type":"title"},"Доверие":{"type":"select"},
				              "День":{"type":"relation"},"Замечено":{"type":"date"}}}`)})
		case "page":
			writeList(w, []json.RawMessage{json.RawMessage(`{
				"object":"page","id":"loose-1","created_time":"2026-08-01T00:00:00.000Z",
				"last_edited_time":"2026-09-10T00:00:00.000Z","archived":false,"in_trash":false,
				"url":"https://app.notion.com/p/loose-1",
				"parent":{"type":"page_id","page_id":"root"},
				"properties":{"title":{"type":"title","title":[{"type":"text","plain_text":"Регламент проекта","annotations":{}}]}}}`)})
		default:
			writeList(w, nil)
		}
	})

	mux.HandleFunc("/v1/databases/db-facts", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"object":"database","id":"db-facts",
			"title":[{"type":"text","plain_text":"🧩 Заметки","annotations":{}}],
			"parent":{"type":"page_id","page_id":"root"},
			"data_sources":[{"id":"ds-facts","name":"🧩 Заметки"}]}`))
	})

	mux.HandleFunc("/v1/data_sources/ds-facts/query", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.mu.Lock()
		f.queries++
		if filter, ok := body["filter"].(map[string]any); ok {
			f.lastFilter = filter
		} else {
			f.lastFilter = nil
		}
		out := make([]json.RawMessage, 0, len(f.rowOrder))
		for _, id := range f.rowOrder {
			out = append(out, f.rows[id])
		}
		f.mu.Unlock()
		writeList(w, out)
	})

	mux.HandleFunc("/v1/blocks/", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/v1/blocks/"), "/children")
		f.mu.Lock()
		blocks := f.blocks[id]
		f.mu.Unlock()
		writeList(w, blocks)
	})

	mux.HandleFunc("/v1/users", func(w http.ResponseWriter, r *http.Request) {
		writeList(w, []json.RawMessage{json.RawMessage(
			`{"object":"user","id":"u1","name":"Ada Lovelace","type":"person","person":{"email":"ada@example.com"}}`)})
	})

	return mux
}

func writeList(w http.ResponseWriter, results []json.RawMessage) {
	if results == nil {
		results = []json.RawMessage{}
	}
	payload, _ := json.Marshal(map[string]any{
		"object": "list", "results": results, "has_more": false, "next_cursor": nil,
	})
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(payload)
}

func newSyncer(t *testing.T, f *fakeNotion) (*Syncer, *mirror.DB, *index.Store) {
	t.Helper()
	srv := httptest.NewServer(f.handler())
	t.Cleanup(srv.Close)

	client, err := notion.New(notion.Options{
		Token: "secret_test", BaseURL: srv.URL + "/v1", RequestsPerSecond: 1000,
	})
	if err != nil {
		t.Fatal(err)
	}
	db, err := mirror.Open(context.Background(), filepath.Join(t.TempDir(), "mirror.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	idx := index.New(db, nil)
	return New(client, db, idx, Options{Logf: t.Logf}), db, idx
}

func baseFake() *fakeNotion {
	return &fakeNotion{
		rows: map[string]json.RawMessage{
			"fact-1": row("fact-1", "Контейнер gateway держит 3,1 ГБ ОЗУ", "2026-09-04T14:47:00.000Z",
				`"Доверие":{"type":"select","select":{"name":"подтверждено"}},"День":{"type":"relation","relation":[{"id":"loose-1"}]}`),
			"fact-2": row("fact-2", "Полив теплицы идёт по датчику", "2026-08-20T10:00:00.000Z",
				`"Доверие":{"type":"select","select":{"name":"вероятно"}}`),
		},
		rowOrder: []string{"fact-1", "fact-2"},
		blocks: map[string][]json.RawMessage{
			"fact-1": {json.RawMessage(`{"object":"block","id":"b1","type":"paragraph","has_children":false,
				"last_edited_time":"2026-09-04T14:47:00.000Z",
				"paragraph":{"rich_text":[{"type":"text","plain_text":"Замер 04.09.2026: usage 3 271 659 520 байт.","annotations":{}}]}}`)},
		},
	}
}

func TestBootstrapMirrorsSchemaRowsAndContent(t *testing.T) {
	ctx := context.Background()
	f := baseFake()
	s, db, _ := newSyncer(t, f)

	sources, err := s.Discover(ctx)
	if err != nil {
		t.Fatalf("discover: %v", err)
	}
	if sources != 1 {
		t.Fatalf("discovered %d data sources", sources)
	}

	st, err := s.Pass(ctx, true)
	if err != nil {
		t.Fatalf("pass: %v", err)
	}
	if st.Pages < 3 {
		t.Fatalf("stats = %+v, expected two rows and one loose page", st)
	}

	// The mirrored data source must answer SQL under its collection:// name.
	var confidence string
	if err := db.SQL().QueryRowContext(ctx,
		`SELECT "Доверие" FROM "collection://ds-facts" WHERE id='fact-1'`).Scan(&confidence); err != nil {
		t.Fatalf("query view: %v", err)
	}
	if confidence != "подтверждено" {
		t.Fatalf("Доверие = %q", confidence)
	}

	page, err := db.Page(ctx, "fact-1")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(page.Markdown, "3 271 659 520") {
		t.Fatalf("block content not rendered: %q", page.Markdown)
	}

	// The relation edge must exist for graph expansion.
	var to string
	if err := db.SQL().QueryRowContext(ctx,
		`SELECT to_id FROM relations WHERE from_id='fact-1'`).Scan(&to); err != nil {
		t.Fatal(err)
	}
	if to != "loose-1" {
		t.Fatalf("edge target = %q", to)
	}
}

func TestIncrementalPassUsesWatermarkFilter(t *testing.T) {
	ctx := context.Background()
	f := baseFake()
	s, _, _ := newSyncer(t, f)

	if _, err := s.Discover(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Pass(ctx, true); err != nil {
		t.Fatal(err)
	}

	f.mu.Lock()
	f.lastFilter = nil
	f.mu.Unlock()

	if _, err := s.Pass(ctx, false); err != nil {
		t.Fatal(err)
	}
	f.mu.Lock()
	filter := f.lastFilter
	f.mu.Unlock()
	if filter == nil {
		t.Fatal("the incremental pass must send a last_edited_time filter")
	}
	if filter["timestamp"] != "last_edited_time" {
		t.Fatalf("filter = %+v", filter)
	}
	inner, ok := filter["last_edited_time"].(map[string]any)
	if !ok || inner["on_or_after"] == "" {
		t.Fatalf("filter body = %+v", filter)
	}
	// Slack must push the bound below the newest edit so an edit inside the
	// same minute is not skipped.
	if got := inner["on_or_after"].(string); got >= "2026-09-04T14:47:00Z" {
		t.Fatalf("watermark %s has no slack", got)
	}
}

func TestReconciliationTrashesRemovedRows(t *testing.T) {
	ctx := context.Background()
	f := baseFake()
	s, db, _ := newSyncer(t, f)
	if _, err := s.Discover(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Pass(ctx, true); err != nil {
		t.Fatal(err)
	}

	f.mu.Lock()
	delete(f.rows, "fact-2")
	f.rowOrder = []string{"fact-1"}
	f.mu.Unlock()

	st, err := s.Pass(ctx, true)
	if err != nil {
		t.Fatal(err)
	}
	if st.Trashed != 1 {
		t.Fatalf("trashed = %d, want 1", st.Trashed)
	}
	page, err := db.Page(ctx, "fact-2")
	if err != nil {
		t.Fatal(err)
	}
	if !page.InTrash {
		t.Fatal("the removed row must be marked trashed, not deleted")
	}
	// History is kept: the row is still readable, just filtered out of views.
	var visible int
	if err := db.SQL().QueryRowContext(ctx,
		`SELECT COUNT(*) FROM "collection://ds-facts" WHERE id='fact-2'`).Scan(&visible); err != nil {
		t.Fatal(err)
	}
	if visible != 0 {
		t.Fatal("a trashed row must not appear in the collection view")
	}
}

func TestContentRefetchedOnlyWhenEditTimeMoves(t *testing.T) {
	ctx := context.Background()
	f := baseFake()
	s, _, _ := newSyncer(t, f)
	if _, err := s.Discover(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Pass(ctx, true); err != nil {
		t.Fatal(err)
	}

	second, err := s.Pass(ctx, false)
	if err != nil {
		t.Fatal(err)
	}
	if second.Content != 0 {
		t.Fatalf("content refetched %d times with no edits", second.Content)
	}

	f.mu.Lock()
	f.rows["fact-1"] = row("fact-1", "Контейнер gateway держит 3,1 ГБ ОЗУ", "2026-09-12T09:00:00.000Z", "")
	f.blocks["fact-1"] = []json.RawMessage{json.RawMessage(`{"object":"block","id":"b1","type":"paragraph","has_children":false,
		"paragraph":{"rich_text":[{"type":"text","plain_text":"Новый замер.","annotations":{}}]}}`)}
	f.mu.Unlock()

	third, err := s.Pass(ctx, false)
	if err != nil {
		t.Fatal(err)
	}
	if third.Content != 1 {
		t.Fatalf("content fetched %d times after an edit, want 1", third.Content)
	}
}

func TestSyncUsers(t *testing.T) {
	ctx := context.Background()
	f := baseFake()
	s, db, _ := newSyncer(t, f)
	if err := s.SyncUsers(ctx); err != nil {
		t.Fatal(err)
	}
	var name string
	if err := db.SQL().QueryRowContext(ctx, `SELECT name FROM users WHERE id='u1'`).Scan(&name); err != nil {
		t.Fatal(err)
	}
	if name != "Ada Lovelace" {
		t.Fatalf("user name = %q", name)
	}
}
