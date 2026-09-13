package mirror

import (
	"strings"
	"testing"
)

const prefsSchema = `{
  "Правило":{"type":"title"},
  "Область":{"type":"select"},
  "Замечено":{"type":"date"},
  "Ядро":{"type":"checkbox"},
  "url":{"type":"url"}
}`

func TestSchemaColumnsExpandsDatesAndEscapesReserved(t *testing.T) {
	cols, err := SchemaColumns(prefsSchema)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{
		"Правило": true, "Область": true, "Ядро": true,
		"date:Замечено:start": true, "date:Замечено:end": true, "date:Замечено:is_datetime": true,
		"userDefined:url": true,
	}
	if len(cols) != len(want) {
		t.Fatalf("got %d columns %v", len(cols), cols)
	}
	for _, c := range cols {
		if !want[c] {
			t.Errorf("unexpected column %q", c)
		}
	}
}

func TestViewSQLUsesCollectionURIAsTableName(t *testing.T) {
	sql, err := ViewSQL("7c1e4a90-5b21-4d8e-9f03-2ab6c7d45e11", prefsSchema)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sql, `CREATE VIEW "collection://7c1e4a90-5b21-4d8e-9f03-2ab6c7d45e11"`) {
		t.Fatalf("view name wrong:\n%s", sql)
	}
	if !strings.Contains(sql, `AS "Правило"`) || !strings.Contains(sql, `AS "date:Замечено:start"`) {
		t.Fatalf("property columns missing:\n%s", sql)
	}
	// A property named url is escaped to userDefined:url, so the row-level
	// url column survives next to it — same split the official server makes.
	if strings.Count(sql, `AS "url"`) != 1 {
		t.Fatalf("row-level url column expected exactly once:\n%s", sql)
	}
	if !strings.Contains(sql, `AS "userDefined:url"`) {
		t.Fatalf("escaped property column missing:\n%s", sql)
	}
}

func TestQuotingEscapesDelimiters(t *testing.T) {
	if got := quoteIdent(`a"b`); got != `"a""b"` {
		t.Fatalf("quoteIdent = %s", got)
	}
	if got := quoteString("it's"); got != `'it''s'` {
		t.Fatalf("quoteString = %s", got)
	}
}
