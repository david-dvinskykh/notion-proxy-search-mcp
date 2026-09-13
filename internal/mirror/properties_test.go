package mirror

import (
	"encoding/json"
	"testing"
)

func flatten(t *testing.T, body string) map[string]any {
	t.Helper()
	cols, _, err := FlattenProperties(json.RawMessage(body))
	if err != nil {
		t.Fatalf("flatten: %v", err)
	}
	got := make(map[string]any, len(cols))
	for _, c := range cols {
		got[c.Name] = c.Value
	}
	return got
}

func TestDatePropertyOnlyHasPrefixedColumns(t *testing.T) {
	got := flatten(t, `{"Замечено":{"type":"date","date":{"start":"2026-09-12","end":null}}}`)
	if _, bare := got["Замечено"]; bare {
		t.Fatal("a date property must not produce a bare column: the official server has none either")
	}
	if got["date:Замечено:start"] != "2026-09-12" {
		t.Fatalf("start column = %v", got["date:Замечено:start"])
	}
	if got["date:Замечено:is_datetime"] != 0.0 {
		t.Fatalf("is_datetime = %v, want 0 for a plain date", got["date:Замечено:is_datetime"])
	}
	if got["date:Замечено:end"] != nil {
		t.Fatalf("end = %v, want nil", got["date:Замечено:end"])
	}
}

func TestDateTimeFlagSetForTimestamps(t *testing.T) {
	got := flatten(t, `{"Когда":{"type":"date","date":{"start":"2026-09-12T08:30:00.000+02:00"}}}`)
	if got["date:Когда:is_datetime"] != 1.0 {
		t.Fatalf("is_datetime = %v, want 1", got["date:Когда:is_datetime"])
	}
}

func TestCheckboxUsesSentinelStrings(t *testing.T) {
	got := flatten(t, `{"Ядро":{"type":"checkbox","checkbox":true},"Спит":{"type":"checkbox","checkbox":false}}`)
	if got["Ядро"] != "__YES__" || got["Спит"] != "__NO__" {
		t.Fatalf("checkbox columns = %v / %v", got["Ядро"], got["Спит"])
	}
}

func TestMultiSelectIsJSONArray(t *testing.T) {
	got := flatten(t, `{"Теги":{"type":"multi_select","multi_select":[{"name":"idea"},{"name":"person"}]}}`)
	if got["Теги"] != `["idea","person"]` {
		t.Fatalf("multi_select = %v", got["Теги"])
	}
}

func TestRelationYieldsEdgesAndJSONColumn(t *testing.T) {
	cols, rels, err := FlattenProperties(json.RawMessage(
		`{"День":{"type":"relation","relation":[{"id":"aaa"},{"id":"bbb"}]}}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(cols) != 1 || cols[0].Value != `["aaa","bbb"]` {
		t.Fatalf("relation column = %+v", cols)
	}
	if len(rels) != 2 || rels[0].Prop != "День" || rels[1].To != "bbb" {
		t.Fatalf("relation edges = %+v", rels)
	}
}

func TestReservedNamesArePrefixed(t *testing.T) {
	got := flatten(t, `{"URL":{"type":"url","url":"https://example.com"},"id":{"type":"number","number":3}}`)
	if got["userDefined:URL"] != "https://example.com" {
		t.Fatalf("URL column = %v", got)
	}
	if got["userDefined:id"] != 3.0 {
		t.Fatalf("id column = %v", got)
	}
}

func TestRollupCarriesComputedValue(t *testing.T) {
	got := flatten(t, `{"Кол-во фактов":{"type":"rollup","rollup":{"type":"number","number":12}}}`)
	if got["Кол-во фактов"] != 12.0 {
		t.Fatalf("rollup = %v, want the number itself", got["Кол-во фактов"])
	}
}

func TestFormulaUnwrapsTypedValue(t *testing.T) {
	got := flatten(t, `{"Частота":{"type":"formula","formula":{"type":"number","number":0.3}}}`)
	if got["Частота"] != 0.3 {
		t.Fatalf("formula = %v", got["Частота"])
	}
}

func TestUniqueIDKeepsPrefix(t *testing.T) {
	got := flatten(t, `{"Номер":{"type":"unique_id","unique_id":{"prefix":"DEC","number":66}}}`)
	if got["Номер"] != "DEC-66" {
		t.Fatalf("unique_id = %v", got["Номер"])
	}
}

func TestUnknownTypeKeepsRawJSON(t *testing.T) {
	got := flatten(t, `{"Новое":{"type":"place","place":{"name":"Kraków"}}}`)
	s, ok := got["Новое"].(string)
	if !ok || s == "" {
		t.Fatalf("unknown property dropped: %v", got["Новое"])
	}
}
