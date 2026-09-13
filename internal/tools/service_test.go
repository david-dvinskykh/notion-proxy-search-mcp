package tools

import (
	"context"
	"database/sql"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/index"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mcp"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mirror"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/notion"
)

func newService(t *testing.T) *Service {
	t.Helper()
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "mirror.db")
	db, err := mirror.Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })

	ro, err := sql.Open("sqlite", "file:"+path+"?mode=ro&_pragma=query_only(1)")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ro.Close() })

	idx := index.New(db, nil)
	svc := &Service{DB: db, RO: ro, Index: idx, Logf: t.Logf}
	seed(t, db, idx)
	return svc
}

func seed(t *testing.T, db *mirror.DB, idx *index.Store) {
	t.Helper()
	ctx := context.Background()

	ds := &notion.DataSource{Schema: json.RawMessage(`{
		"Утверждение":{"type":"title"},"Доверие":{"type":"select"},
		"День":{"type":"relation"},"Замечено":{"type":"date"}}`)}
	ds.ID = "ds-facts"
	ds.Title = []notion.RichText{{PlainText: "🧩 Заметки"}}
	if err := db.UpsertDataSource(ctx, ds); err != nil {
		t.Fatal(err)
	}
	journal := &notion.DataSource{Schema: json.RawMessage(`{"Дата":{"type":"title"}}`)}
	journal.ID = "ds-journal"
	journal.Title = []notion.RichText{{PlainText: "🗓️ Дневник"}}
	if err := db.UpsertDataSource(ctx, journal); err != nil {
		t.Fatal(err)
	}

	add := func(id, dsID, props, markdown, edited string) {
		p := &notion.Page{}
		p.ID = id
		p.Object.Object = "page"
		p.URL = "https://app.notion.com/p/" + id
		p.LastEditedTime = edited
		p.Icon = json.RawMessage(`{"type":"emoji","emoji":"🧩"}`)
		p.Properties = json.RawMessage(props)
		if _, err := db.UpsertPage(ctx, p, dsID); err != nil {
			t.Fatal(err)
		}
		if err := db.SetContent(ctx, id, markdown); err != nil {
			t.Fatal(err)
		}
		if err := idx.Reindex(ctx, id); err != nil {
			t.Fatal(err)
		}
	}

	add("day-1", "ds-journal", `{"Дата":{"type":"title","title":[{"type":"text","plain_text":"2026-09-04","annotations":{}}]}}`,
		"Карточка дня.\n", "2026-09-04T21:00:00.000Z")
	add("fact-ram", "ds-facts", `{
		"Утверждение":{"type":"title","title":[{"type":"text","plain_text":"Контейнер gateway держит 3,1 ГБ ОЗУ","annotations":{}}]},
		"Доверие":{"type":"select","select":{"name":"подтверждено"}},
		"Замечено":{"type":"date","date":{"start":"2026-09-04"}},
		"День":{"type":"relation","relation":[{"id":"day-1"}]}}`,
		"Замер 04.09.2026: usage 3 271 659 520 байт при лимите 8 ГБ.\n", "2026-09-04T14:47:00.000Z")
}

func TestQueryRunsConstitutionStyleSQL(t *testing.T) {
	svc := newService(t)
	got, err := svc.Query(context.Background(), QueryArgs{
		Query:  `SELECT url,"Утверждение","Доверие" FROM "collection://ds-facts" WHERE "Доверие" = ?`,
		Params: []any{"подтверждено"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.RowCount != 1 {
		t.Fatalf("rows = %d", got.RowCount)
	}
	if got.Columns[1] != "Утверждение" {
		t.Fatalf("columns = %v", got.Columns)
	}
}

func TestQueryCanJoinTwoDataSources(t *testing.T) {
	svc := newService(t)
	got, err := svc.Query(context.Background(), QueryArgs{Query: `
		SELECT f."Утверждение", j."Дата"
		FROM "collection://ds-facts" f
		JOIN relations r ON r.from_id = f.id
		JOIN "collection://ds-journal" j ON j.id = r.to_id`})
	if err != nil {
		t.Fatalf("multi-source join failed: %v", err)
	}
	if got.RowCount != 1 {
		t.Fatalf("rows = %d, want the joined pair", got.RowCount)
	}
}

func TestQueryRejectsWrites(t *testing.T) {
	svc := newService(t)
	for _, statement := range []string{
		`DELETE FROM pages`,
		`UPDATE pages SET title='x'`,
		`PRAGMA writable_schema=1`,
		`SELECT 1; DELETE FROM pages`,
		`DROP VIEW "collection://ds-facts"`,
	} {
		if _, err := svc.Query(context.Background(), QueryArgs{Query: statement}); err == nil {
			t.Errorf("statement %q was accepted", statement)
		}
	}
}

func TestQueryTruncatesLargeResults(t *testing.T) {
	svc := newService(t)
	got, err := svc.Query(context.Background(), QueryArgs{
		Query: `WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x < 50) SELECT x FROM n`,
		Limit: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.RowCount != 10 || !got.Truncated {
		t.Fatalf("rows = %d truncated = %v", got.RowCount, got.Truncated)
	}
}

func TestFetchRendersPropertiesAndContent(t *testing.T) {
	svc := newService(t)
	out, err := svc.Fetch(context.Background(), FetchArgs{ID: "fact-ram", IncludeRelations: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`<page id="fact-ram"`,
		`table="collection://ds-facts"`,
		`"Доверие": "подтверждено"`,
		`"date:Замечено:start": "2026-09-04"`,
		"3 271 659 520",
		"День: 2026-09-04",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("fetch output lacks %q:\n%s", want, out)
		}
	}
}

func TestFetchAcceptsNotionURL(t *testing.T) {
	svc := newService(t)
	if _, err := svc.Fetch(context.Background(),
		FetchArgs{ID: "https://app.notion.com/p/fact-ram?pvs=204"}); err != nil {
		t.Fatalf("URL form rejected: %v", err)
	}
}

func TestFetchUnknownPageExplainsItself(t *testing.T) {
	svc := newService(t)
	_, err := svc.Fetch(context.Background(), FetchArgs{ID: "missing-page"})
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "not in the mirror") {
		t.Fatalf("unhelpful error: %v", err)
	}
}

func TestListDataSourcesReportsTablesAndColumns(t *testing.T) {
	svc := newService(t)
	got, err := svc.ListDataSources(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d data sources", len(got))
	}
	var facts *DataSourceInfo
	for i := range got {
		if got[i].ID == "ds-facts" {
			facts = &got[i]
		}
	}
	if facts == nil {
		t.Fatal("facts data source missing")
	}
	if facts.Table != `collection://ds-facts` || facts.Rows != 1 {
		t.Fatalf("catalog entry = %+v", facts)
	}
	joined := strings.Join(facts.Columns, ",")
	if !strings.Contains(joined, "date:Замечено:start") || !strings.Contains(joined, "Доверие") {
		t.Fatalf("columns = %v", facts.Columns)
	}
}

func TestSearchWithoutModelReportsDegradation(t *testing.T) {
	svc := newService(t)
	got, err := svc.Search(context.Background(), SearchArgs{Query: "gateway ОЗУ"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Mode != "keyword" {
		t.Fatalf("mode = %q", got.Mode)
	}
	if got.Degraded == "" {
		t.Fatal("a hybrid request answered without the model must say so")
	}
	if got.Count == 0 {
		t.Fatal("keyword fallback returned nothing")
	}
}

func TestRecallGroupsLinkedPagesBySource(t *testing.T) {
	svc := newService(t)
	got, err := svc.Recall(context.Background(), RecallArgs{Entity: "gateway ОЗУ"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Entity == nil || got.Entity.PageID != "fact-ram" {
		t.Fatalf("entity = %+v", got.Entity)
	}
	linked, ok := got.Linked["🗓️ Дневник"]
	if !ok || len(linked) != 1 || linked[0].ID != "day-1" {
		t.Fatalf("linked = %+v", got.Linked)
	}
}

func TestStatusReportsCountsAndServer(t *testing.T) {
	svc := newService(t)
	got, err := svc.Status(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	counts := got["counts"].(map[string]int)
	if counts["pages"] != 2 {
		t.Fatalf("counts = %+v", counts)
	}
	if got["embedding_model"] != "none (keyword search only)" {
		t.Fatalf("embedding_model = %v", got["embedding_model"])
	}
}

func TestResyncWithoutDaemonFailsClearly(t *testing.T) {
	svc := newService(t)
	_, err := svc.Resync(context.Background(), ResyncArgs{PageID: "fact-ram"})
	if err == nil || !strings.Contains(err.Error(), "no Notion credentials") {
		t.Fatalf("error = %v", err)
	}
}

// TestToolsAreReachableOverMCP drives the registered tools through the real
// protocol, which is what an MCP gateway does.
func TestToolsAreReachableOverMCP(t *testing.T) {
	svc := newService(t)
	srv := mcp.NewServer("notion-proxy-search", "test", nil)
	svc.Register(srv)

	requests := []string{
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}`,
		`{"jsonrpc":"2.0","id":2,"method":"tools/list"}`,
		`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"notion-list-data-sources","arguments":{}}}`,
		`{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"notion-search","arguments":{"query":"ОЗУ","mode":"keyword"}}}`,
		`{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"notion-fetch","arguments":{"id":"fact-ram"}}}`,
		`{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"notion-query-data-sources","arguments":{"query":"SELECT COUNT(*) AS n FROM pages"}}}`,
	}
	var out strings.Builder
	if err := srv.Serve(context.Background(), strings.NewReader(strings.Join(requests, "\n")+"\n"), &out); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != len(requests) {
		t.Fatalf("got %d replies for %d requests", len(lines), len(requests))
	}
	for i, line := range lines {
		var reply map[string]any
		if err := json.Unmarshal([]byte(line), &reply); err != nil {
			t.Fatalf("reply %d: %v", i, err)
		}
		if reply["error"] != nil {
			t.Fatalf("request %d failed: %v", i+1, reply["error"])
		}
		if result, ok := reply["result"].(map[string]any); ok && result["isError"] == true {
			content := result["content"].([]any)[0].(map[string]any)
			t.Fatalf("request %d returned a tool error: %v", i+1, content["text"])
		}
	}

	// The tool list must advertise every tool with a schema.
	var listReply map[string]any
	if err := json.Unmarshal([]byte(lines[1]), &listReply); err != nil {
		t.Fatal(err)
	}
	tools := listReply["result"].(map[string]any)["tools"].([]any)
	if len(tools) != 7 {
		t.Fatalf("advertised %d tools, want 7", len(tools))
	}
	names := map[string]bool{}
	for _, raw := range tools {
		tool := raw.(map[string]any)
		names[tool["name"].(string)] = true
		if tool["description"] == "" {
			t.Fatalf("tool %v has no description", tool["name"])
		}
		if _, ok := tool["inputSchema"].(map[string]any); !ok {
			t.Fatalf("tool %v has no input schema", tool["name"])
		}
	}
	for _, want := range []string{
		"notion-search", "notion-fetch", "notion-query-data-sources",
		"notion-list-data-sources", "notion-recall", "notion-mirror-status", "notion-resync",
	} {
		if !names[want] {
			t.Errorf("tool %s is not advertised", want)
		}
	}
}
