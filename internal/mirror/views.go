package mirror

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// fixedColumns are the row-level columns every collection view exposes in
// addition to the data source's own properties. A property with the same name
// wins, so a workspace that has a property called "title" is not shadowed.
var fixedColumns = []struct{ alias, expr string }{
	{"id", "p.id"},
	{"url", "p.url"},
	{"title", "p.title"},
	{"icon", "p.icon"},
	{"created_time", "p.created_time"},
	{"last_edited_time", "p.last_edited_time"},
	{"content_markdown", "p.content_markdown"},
	{"parent_id", "p.parent_id"},
}

// SchemaColumns returns the SQL column names a data source schema produces, in
// a stable order.
func SchemaColumns(schemaJSON string) ([]string, error) {
	var schema map[string]struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal([]byte(schemaJSON), &schema); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(schema))
	for name := range schema {
		names = append(names, name)
	}
	sort.Strings(names)

	cols := make([]string, 0, len(names)+2)
	for _, name := range names {
		if schema[name].Type == "date" {
			for _, c := range DateColumns(name) {
				cols = append(cols, c)
			}
			continue
		}
		cols = append(cols, ColumnName(name))
	}
	return cols, nil
}

// ViewSQL builds the CREATE VIEW statement for one data source. The view is
// named exactly "collection://<data source id>" so SQL written for the
// official server's query tool runs here without edits.
func ViewSQL(dataSourceID, schemaJSON string) (string, error) {
	cols, err := SchemaColumns(schemaJSON)
	if err != nil {
		return "", err
	}
	taken := make(map[string]bool, len(cols))
	for _, c := range cols {
		taken[c] = true
	}

	var selects []string
	for _, fc := range fixedColumns {
		if taken[fc.alias] {
			continue
		}
		selects = append(selects, fmt.Sprintf("%s AS %s", fc.expr, quoteIdent(fc.alias)))
	}
	for _, c := range cols {
		selects = append(selects, fmt.Sprintf(
			"MAX(CASE WHEN v.col=%s THEN v.value END) AS %s", quoteString(c), quoteIdent(c)))
	}

	return fmt.Sprintf(`CREATE VIEW %s AS
SELECT %s
FROM pages p LEFT JOIN page_props v ON v.page_id = p.id
WHERE p.data_source_id = %s AND p.in_trash = 0 AND p.archived = 0
GROUP BY p.id`,
		quoteIdent(ViewName(dataSourceID)),
		strings.Join(selects, ",\n       "),
		quoteString(dataSourceID)), nil
}

// ViewName is the table name a query uses for a data source.
func ViewName(dataSourceID string) string { return "collection://" + dataSourceID }

func quoteIdent(s string) string { return `"` + strings.ReplaceAll(s, `"`, `""`) + `"` }

func quoteString(s string) string { return `'` + strings.ReplaceAll(s, `'`, `''`) + `'` }
