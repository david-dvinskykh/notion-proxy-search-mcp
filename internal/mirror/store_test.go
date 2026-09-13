package mirror

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/notion"
)

func openTestDB(t *testing.T) *DB {
	t.Helper()
	db, err := Open(context.Background(), filepath.Join(t.TempDir(), "mirror.db"))
	if err != nil {
		t.Fatalf("open mirror: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func mustUpsertSource(t *testing.T, db *DB, id, schema string) {
	t.Helper()
	ds := &notion.DataSource{Schema: json.RawMessage(schema)}
	ds.ID = id
	ds.Title = []notion.RichText{{PlainText: "🎛️ Правила"}}
	if err := db.UpsertDataSource(context.Background(), ds); err != nil {
		t.Fatalf("upsert data source: %v", err)
	}
}

func mustUpsertPage(t *testing.T, db *DB, id, dsID, props string) bool {
	t.Helper()
	p := &notion.Page{}
	p.ID = id
	p.Object.Object = "page"
	p.URL = "https://app.notion.com/" + id
	p.LastEditedTime = "2026-09-12T06:00:00.000Z"
	p.Properties = json.RawMessage(props)
	changed, err := db.UpsertPage(context.Background(), p, dsID)
	if err != nil {
		t.Fatalf("upsert page: %v", err)
	}
	return changed
}

// TestCollectionViewAnswersTheConstitutionQuery runs the exact SQL the memory
// constitution uses against the mirror. It is the compatibility contract: if
// this breaks, queries written for the official server break too.
func TestCollectionViewAnswersTheConstitutionQuery(t *testing.T) {
	db := openTestDB(t)
	const dsID = "7c1e4a90-5b21-4d8e-9f03-2ab6c7d45e11"
	mustUpsertSource(t, db, dsID, `{
		"Правило":{"type":"title"},"Область":{"type":"select"},"Триггер":{"type":"rich_text"},
		"Сила":{"type":"select"},"Статус":{"type":"select"},"Нарушено":{"type":"date"}}`)

	mustUpsertPage(t, db, "rule-1", dsID, `{
		"Правило":{"type":"title","title":[{"type":"text","plain_text":"Отвечать по сути","annotations":{}}]},
		"Область":{"type":"select","select":{"name":"ответы и формат"}},
		"Триггер":{"type":"rich_text","rich_text":[{"type":"text","plain_text":"любой ответ","annotations":{}}]},
		"Сила":{"type":"select","select":{"name":"закон"}},
		"Статус":{"type":"select","select":{"name":"действует"}},
		"Нарушено":{"type":"date","date":null}}`)
	mustUpsertPage(t, db, "rule-2", dsID, `{
		"Правило":{"type":"title","title":[{"type":"text","plain_text":"Спящее правило","annotations":{}}]},
		"Область":{"type":"select","select":{"name":"автономность"}},
		"Статус":{"type":"select","select":{"name":"уснуло"}}}`)

	query := `SELECT url,"Правило","Область","Триггер","Сила" FROM "collection://` + dsID +
		`" WHERE "Статус" = 'действует' ORDER BY "Область"`
	rows, err := db.SQL().QueryContext(context.Background(), query)
	if err != nil {
		t.Fatalf("constitution query failed: %v", err)
	}
	defer rows.Close()

	var got int
	for rows.Next() {
		var url, rule, area string
		var trigger, strength *string
		if err := rows.Scan(&url, &rule, &area, &trigger, &strength); err != nil {
			t.Fatalf("scan: %v", err)
		}
		got++
		if rule != "Отвечать по сути" || area != "ответы и формат" {
			t.Fatalf("unexpected row %q / %q", rule, area)
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Fatalf("got %d rows, want only the active rule", got)
	}
}

func TestDateColumnsAreQueryableInSQL(t *testing.T) {
	db := openTestDB(t)
	const dsID = "3f60b8d2-91ac-4e57-8b14-6de0a2f73c95"
	mustUpsertSource(t, db, dsID, `{"Дата":{"type":"title"},"Создано":{"type":"date"}}`)
	mustUpsertPage(t, db, "day-1", dsID, `{
		"Дата":{"type":"title","title":[{"type":"text","plain_text":"2026-09-12","annotations":{}}]},
		"Создано":{"type":"date","date":{"start":"2026-09-12"}}}`)

	var n int
	err := db.SQL().QueryRowContext(context.Background(),
		`SELECT COUNT(*) FROM "collection://`+dsID+`" WHERE "date:Создано:start" >= '2026-09-01'`).Scan(&n)
	if err != nil {
		t.Fatalf("date query: %v", err)
	}
	if n != 1 {
		t.Fatalf("got %d rows", n)
	}
}

func TestUpsertPageReportsChangeOnlyWhenEditedTimeMoves(t *testing.T) {
	db := openTestDB(t)
	const dsID = "ds-1"
	mustUpsertSource(t, db, dsID, `{"Name":{"type":"title"}}`)
	props := `{"Name":{"type":"title","title":[{"type":"text","plain_text":"x","annotations":{}}]}}`

	if !mustUpsertPage(t, db, "p1", dsID, props) {
		t.Fatal("first insert must report a change")
	}
	if mustUpsertPage(t, db, "p1", dsID, props) {
		t.Fatal("unchanged last_edited_time must not report a change")
	}
}

func TestSchemaChangeRebuildsView(t *testing.T) {
	db := openTestDB(t)
	const dsID = "ds-2"
	mustUpsertSource(t, db, dsID, `{"Name":{"type":"title"}}`)
	mustUpsertSource(t, db, dsID, `{"Name":{"type":"title"},"Новое поле":{"type":"rich_text"}}`)

	var n int
	if err := db.SQL().QueryRowContext(context.Background(),
		`SELECT COUNT("Новое поле") FROM "collection://`+dsID+`"`).Scan(&n); err != nil {
		t.Fatalf("new column not visible after schema change: %v", err)
	}
}

func TestRelationsAreStoredForGraphExpansion(t *testing.T) {
	db := openTestDB(t)
	const dsID = "ds-3"
	mustUpsertSource(t, db, dsID, `{"Name":{"type":"title"},"День":{"type":"relation"}}`)
	mustUpsertPage(t, db, "fact-1", dsID, `{
		"Name":{"type":"title","title":[{"type":"text","plain_text":"факт","annotations":{}}]},
		"День":{"type":"relation","relation":[{"id":"day-1"}]}}`)

	var to string
	if err := db.SQL().QueryRowContext(context.Background(),
		`SELECT to_id FROM relations WHERE from_id='fact-1'`).Scan(&to); err != nil {
		t.Fatal(err)
	}
	if to != "day-1" {
		t.Fatalf("edge target = %q", to)
	}
}

func TestEnsureViewsRecreatesAfterReopen(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "mirror.db")
	ctx := context.Background()

	db, err := Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	mustUpsertSource(t, db, "ds-4", `{"Name":{"type":"title"}}`)
	db.Close()

	reopened, err := Open(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if err := reopened.EnsureViews(ctx); err != nil {
		t.Fatalf("ensure views: %v", err)
	}
	var n int
	if err := reopened.SQL().QueryRowContext(ctx, `SELECT COUNT(*) FROM "collection://ds-4"`).Scan(&n); err != nil {
		t.Fatalf("view missing after reopen: %v", err)
	}
}
