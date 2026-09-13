// Package tools implements the operations the MCP tools expose, independent of
// transport: the stdio process and the daemon call the same code.
package tools

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/index"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mirror"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/sync"
)

// Remote is the subset of operations the stdio process forwards to the daemon,
// because they need the embedding model or the Notion token.
type Remote interface {
	Search(ctx context.Context, q index.Query) ([]index.Result, error)
	Resync(ctx context.Context, pageID string, full bool) (string, error)
	Status(ctx context.Context) (map[string]any, error)
}

// Service is the operation set behind the tools.
type Service struct {
	DB     *mirror.DB
	RO     *sql.DB
	Index  *index.Store
	Syncer *sync.Syncer
	Remote Remote
	Logf   func(format string, args ...any)
}

func (s *Service) logf(format string, args ...any) {
	if s.Logf != nil {
		s.Logf(format, args...)
	}
}

// SearchArgs is the input of the search tool.
type SearchArgs struct {
	Query           string   `json:"query"`
	Limit           int      `json:"limit"`
	Mode            string   `json:"mode"`
	DataSources     []string `json:"data_sources"`
	PageIDs         []string `json:"page_ids"`
	EditedAfter     string   `json:"edited_after"`
	EditedBefore    string   `json:"edited_before"`
	ExpandRelations bool     `json:"expand_relations"`
}

// SearchResponse is what the search tool returns.
type SearchResponse struct {
	Mode     string         `json:"mode"`
	Degraded string         `json:"degraded,omitempty"`
	Count    int            `json:"count"`
	Results  []index.Result `json:"results"`
}

// Search answers a retrieval request. Vector and hybrid modes need the model,
// which lives in the daemon; when the daemon is unreachable the call degrades
// to keyword search and says so rather than failing.
func (s *Service) Search(ctx context.Context, args SearchArgs) (*SearchResponse, error) {
	if strings.TrimSpace(args.Query) == "" {
		return nil, fmt.Errorf("search: query is required")
	}
	q := index.Query{
		Text:            args.Query,
		Limit:           args.Limit,
		Mode:            index.Mode(args.Mode),
		DataSources:     args.DataSources,
		PageIDs:         args.PageIDs,
		EditedAfter:     args.EditedAfter,
		EditedBefore:    args.EditedBefore,
		ExpandRelations: args.ExpandRelations,
	}
	if q.Mode == "" {
		q.Mode = index.ModeHybrid
	}

	needsModel := q.Mode == index.ModeHybrid || q.Mode == index.ModeVector
	degraded := ""
	if needsModel && !s.Index.HasEmbedder() {
		if s.Remote != nil {
			results, err := s.Remote.Search(ctx, q)
			if err == nil {
				return &SearchResponse{Mode: string(q.Mode), Count: len(results), Results: results}, nil
			}
			s.logf("search: daemon unreachable (%v), falling back to keyword", err)
			degraded = "the sync daemon is not answering, so only keyword search ran: " + err.Error()
		} else {
			degraded = "no embedding model is loaded, so only keyword search ran"
		}
		q.Mode = index.ModeKeyword
	}

	results, err := s.Index.Search(ctx, q)
	if err != nil {
		return nil, err
	}
	return &SearchResponse{Mode: string(q.Mode), Degraded: degraded, Count: len(results), Results: results}, nil
}

// FetchArgs is the input of the fetch tool.
type FetchArgs struct {
	ID string `json:"id"`
	// IncludeRelations lists the pages this one points at, which is how a fact
	// leads to its entity card without a second call.
	IncludeRelations bool `json:"include_relations"`
}

// Fetch renders one mirrored page the way an agent wants to read it.
func (s *Service) Fetch(ctx context.Context, args FetchArgs) (string, error) {
	id := normalizeID(args.ID)
	if id == "" {
		return "", fmt.Errorf("fetch: id is required")
	}
	page, err := s.lookupPage(ctx, id)
	if err != nil {
		return "", err
	}

	var b strings.Builder
	fmt.Fprintf(&b, "<page id=%q url=%q", page.ID, page.URL)
	if page.Icon != "" {
		fmt.Fprintf(&b, " icon=%q", page.Icon)
	}
	b.WriteString(">\n")
	if page.DataSourceID != "" {
		title, _ := s.dataSourceTitle(ctx, page.DataSourceID)
		fmt.Fprintf(&b, "<data-source id=%q name=%q table=%q/>\n",
			page.DataSourceID, title, mirror.ViewName(page.DataSourceID))
	}
	fmt.Fprintf(&b, "<mirror last_edited_time=%q synced=%q/>\n", page.LastEditedTime, page.ContentSynced)
	if page.InTrash {
		b.WriteString("<trashed>this page is in the Notion trash; the mirror keeps it for history</trashed>\n")
	}

	props, err := s.propertyMap(ctx, page.ID)
	if err != nil {
		return "", err
	}
	if len(props) > 0 {
		encoded, err := json.MarshalIndent(props, "", "  ")
		if err != nil {
			return "", err
		}
		b.WriteString("<properties>\n")
		b.Write(encoded)
		b.WriteString("\n</properties>\n")
	}
	if args.IncludeRelations {
		related, err := s.relatedPages(ctx, page.ID)
		if err != nil {
			return "", err
		}
		if len(related) > 0 {
			b.WriteString("<relations>\n")
			for _, r := range related {
				fmt.Fprintf(&b, "- %s: %s (%s)\n", r.Prop, r.Title, r.URL)
			}
			b.WriteString("</relations>\n")
		}
	}
	b.WriteString("<content>\n")
	b.WriteString(page.Markdown)
	b.WriteString("\n</content>\n</page>\n")
	return b.String(), nil
}

// lookupPage accepts a dashed or dashless id and falls back to a title match,
// so a pasted page name still resolves.
func (s *Service) lookupPage(ctx context.Context, id string) (*mirror.PageRow, error) {
	page, err := s.DB.Page(ctx, id)
	if err == nil {
		return page, nil
	}
	if err != sql.ErrNoRows {
		return nil, err
	}
	var found string
	err = s.RO.QueryRowContext(ctx,
		`SELECT id FROM pages WHERE REPLACE(id,'-','') = REPLACE(?,'-','') LIMIT 1`, id).Scan(&found)
	if err == nil {
		return s.DB.Page(ctx, found)
	}
	if err != sql.ErrNoRows {
		return nil, err
	}
	return nil, fmt.Errorf("fetch: %s is not in the mirror (it may be outside the integration's access, or not synced yet)", id)
}

type relation struct {
	Prop  string
	Title string
	URL   string
}

func (s *Service) relatedPages(ctx context.Context, pageID string) ([]relation, error) {
	rows, err := s.RO.QueryContext(ctx, `
		SELECT r.prop, p.title, p.url FROM relations r
		JOIN pages p ON p.id = r.to_id
		WHERE r.from_id = ? AND p.in_trash = 0
		ORDER BY r.prop, p.title LIMIT 200`, pageID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []relation
	for rows.Next() {
		var r relation
		if err := rows.Scan(&r.Prop, &r.Title, &r.URL); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *Service) propertyMap(ctx context.Context, pageID string) (map[string]any, error) {
	rows, err := s.RO.QueryContext(ctx,
		`SELECT col, value FROM page_props WHERE page_id=? ORDER BY col`, pageID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]any{}
	for rows.Next() {
		var col string
		var value any
		if err := rows.Scan(&col, &value); err != nil {
			return nil, err
		}
		if value == nil {
			continue
		}
		out[col] = value
	}
	return out, rows.Err()
}

func (s *Service) dataSourceTitle(ctx context.Context, id string) (string, error) {
	var title string
	err := s.RO.QueryRowContext(ctx, `SELECT title FROM data_sources WHERE id=?`, id).Scan(&title)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return title, err
}

// QueryArgs is the input of the SQL tool.
type QueryArgs struct {
	Query  string `json:"query"`
	Params []any  `json:"params"`
	Limit  int    `json:"limit"`
}

// QueryResponse holds a result set.
type QueryResponse struct {
	Columns   []string `json:"columns"`
	Rows      [][]any  `json:"rows"`
	RowCount  int      `json:"row_count"`
	Truncated bool     `json:"truncated"`
}

// maxRows caps a result set so a careless SELECT cannot flood the context.
const maxRows = 500

// Query runs read-only SQL against the mirror. The connection is opened
// read-only, so a write cannot succeed even if the statement check is fooled.
func (s *Service) Query(ctx context.Context, args QueryArgs) (*QueryResponse, error) {
	statement := strings.TrimSpace(args.Query)
	if statement == "" {
		return nil, fmt.Errorf("query: a SQL statement is required")
	}
	if err := checkReadOnly(statement); err != nil {
		return nil, err
	}
	limit := args.Limit
	if limit <= 0 || limit > maxRows {
		limit = maxRows
	}

	rows, err := s.RO.QueryContext(ctx, statement, args.Params...)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()
	cols, err := rows.Columns()
	if err != nil {
		return nil, err
	}

	out := &QueryResponse{Columns: cols, Rows: [][]any{}}
	for rows.Next() {
		if len(out.Rows) >= limit {
			out.Truncated = true
			break
		}
		holders := make([]any, len(cols))
		for i := range holders {
			holders[i] = new(any)
		}
		if err := rows.Scan(holders...); err != nil {
			return nil, err
		}
		record := make([]any, len(cols))
		for i, h := range holders {
			record[i] = normalizeValue(*(h.(*any)))
		}
		out.Rows = append(out.Rows, record)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	out.RowCount = len(out.Rows)
	return out, nil
}

// normalizeValue turns driver values into JSON-friendly ones.
func normalizeValue(v any) any {
	switch t := v.(type) {
	case []byte:
		return string(t)
	default:
		return v
	}
}

// forbidden are statement heads that could change the mirror. The read-only
// connection already blocks them; this check exists to give a clear error
// instead of a driver-level one.
var forbidden = []string{
	"insert", "update", "delete", "drop", "alter", "create", "replace",
	"attach", "detach", "vacuum", "reindex", "pragma", "begin", "commit", "rollback",
}

func checkReadOnly(statement string) error {
	lower := strings.ToLower(statement)
	head := strings.TrimLeft(lower, "( \t\n\r")
	if !strings.HasPrefix(head, "select") && !strings.HasPrefix(head, "with") {
		return fmt.Errorf("query: only SELECT and WITH statements are allowed; writes go to Notion through the official MCP server")
	}
	for _, word := range forbidden {
		if strings.HasPrefix(head, word) {
			return fmt.Errorf("query: %s is not allowed here", strings.ToUpper(word))
		}
	}
	// Reject a second statement: "select 1; delete from pages" must not pass.
	if idx := strings.Index(statement, ";"); idx >= 0 && strings.TrimSpace(statement[idx+1:]) != "" {
		return fmt.Errorf("query: one statement per call")
	}
	return nil
}

// DataSourceInfo describes a mirrored data source for the catalog tool.
type DataSourceInfo struct {
	ID         string   `json:"id"`
	Title      string   `json:"title"`
	Table      string   `json:"table"`
	DatabaseID string   `json:"database_id,omitempty"`
	Rows       int      `json:"rows"`
	Columns    []string `json:"columns"`
	Watermark  string   `json:"synced_through,omitempty"`
}

// ListDataSources returns the catalog an agent needs to write SQL without
// fetching each database first.
func (s *Service) ListDataSources(ctx context.Context) ([]DataSourceInfo, error) {
	sources, err := s.DB.DataSources(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]DataSourceInfo, 0, len(sources))
	for _, ds := range sources {
		cols, err := mirror.SchemaColumns(ds.SchemaJSON)
		if err != nil {
			s.logf("catalog: schema for %s: %v", ds.ID, err)
		}
		var rowCount int
		if err := s.RO.QueryRowContext(ctx,
			`SELECT COUNT(*) FROM pages WHERE data_source_id=? AND in_trash=0`, ds.ID).Scan(&rowCount); err != nil {
			return nil, err
		}
		out = append(out, DataSourceInfo{
			ID: ds.ID, Title: ds.Title, Table: mirror.ViewName(ds.ID),
			DatabaseID: ds.DatabaseID, Rows: rowCount, Columns: cols, Watermark: ds.Watermark,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Title < out[j].Title })
	return out, nil
}

// RecallArgs is the input of the recall tool.
type RecallArgs struct {
	Entity string `json:"entity"`
	Limit  int    `json:"limit"`
}

// RecallResponse groups everything the mirror knows about one entity.
type RecallResponse struct {
	Entity  *index.Result           `json:"entity,omitempty"`
	Matches []index.Result          `json:"matches"`
	Linked  map[string][]LinkedPage `json:"linked_by_source,omitempty"`
}

// LinkedPage is one page related to the entity.
type LinkedPage struct {
	ID      string `json:"id"`
	Title   string `json:"title"`
	URL     string `json:"url"`
	Prop    string `json:"via_property"`
	Edited  string `json:"last_edited_time,omitempty"`
	Snippet string `json:"snippet,omitempty"`
}

// Recall is the entity-first read: find the card, then everything pointing at
// it, grouped by the database it lives in. It is one call instead of the search
// plus per-relation fetches the same question otherwise costs.
func (s *Service) Recall(ctx context.Context, args RecallArgs) (*RecallResponse, error) {
	if strings.TrimSpace(args.Entity) == "" {
		return nil, fmt.Errorf("recall: entity is required")
	}
	limit := args.Limit
	if limit <= 0 {
		limit = 8
	}
	found, err := s.Search(ctx, SearchArgs{Query: args.Entity, Limit: limit})
	if err != nil {
		return nil, err
	}
	out := &RecallResponse{Matches: found.Results}
	if len(found.Results) == 0 {
		return out, nil
	}
	best := found.Results[0]
	out.Entity = &best

	rows, err := s.RO.QueryContext(ctx, `
		SELECT r.prop, p.id, p.title, p.url, p.last_edited_time,
		       COALESCE((SELECT ds.title FROM data_sources ds WHERE ds.id = p.data_source_id), 'вне базы') AS source
		FROM relations r
		JOIN pages p ON p.id = CASE WHEN r.from_id = ? THEN r.to_id ELSE r.from_id END
		WHERE (r.from_id = ? OR r.to_id = ?) AND p.in_trash = 0
		ORDER BY source, p.last_edited_time DESC
		LIMIT 200`, best.PageID, best.PageID, best.PageID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	linked := map[string][]LinkedPage{}
	for rows.Next() {
		var lp LinkedPage
		var source string
		if err := rows.Scan(&lp.Prop, &lp.ID, &lp.Title, &lp.URL, &lp.Edited, &source); err != nil {
			return nil, err
		}
		linked[source] = append(linked[source], lp)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(linked) > 0 {
		out.Linked = linked
	}
	return out, nil
}

// Status reports what the mirror holds and how fresh it is.
func (s *Service) Status(ctx context.Context) (map[string]any, error) {
	if s.Syncer == nil && s.Remote != nil {
		if status, err := s.Remote.Status(ctx); err == nil {
			status["served_by"] = "daemon"
			return status, nil
		} else {
			s.logf("status: daemon unreachable: %v", err)
		}
	}
	stats, err := s.Index.Stats(ctx)
	if err != nil {
		return nil, err
	}
	out := map[string]any{"counts": stats}
	if name := s.Index.ModelName(); name != "" {
		out["embedding_model"] = name
	} else {
		out["embedding_model"] = "none (keyword search only)"
	}
	if skipped := s.Index.Skipped(); len(skipped) > 0 {
		out["not_embedded"] = skipped
	}
	for _, key := range []string{"last_pass", "last_full_pass", "loose_watermark", "bootstrapped"} {
		if v, err := s.DB.Meta(ctx, key); err == nil && v != "" {
			out[key] = v
		}
	}
	if lines, err := s.DB.RecentLog(ctx, 10); err == nil {
		out["recent"] = lines
	}
	if s.Syncer != nil {
		out["served_by"] = "daemon"
	} else {
		out["served_by"] = "stdio process (read-only mirror)"
	}
	return out, nil
}

// ResyncArgs is the input of the resync tool.
type ResyncArgs struct {
	PageID string `json:"page_id"`
	Full   bool   `json:"full"`
}

// Resync pulls fresh data from Notion now instead of at the next poll, which is
// what an agent wants right after writing through the official server.
func (s *Service) Resync(ctx context.Context, args ResyncArgs) (string, error) {
	if s.Syncer == nil {
		if s.Remote == nil {
			return "", fmt.Errorf("resync: this process has no Notion credentials and no daemon to ask")
		}
		return s.Remote.Resync(ctx, args.PageID, args.Full)
	}
	if args.PageID != "" {
		id := normalizeID(args.PageID)
		if err := s.Syncer.ResyncPage(ctx, id); err != nil {
			return "", err
		}
		return fmt.Sprintf("re-read %s from Notion", id), nil
	}
	st, err := s.Syncer.Pass(ctx, args.Full)
	if err != nil {
		return "", err
	}
	return "sync pass: " + st.Describe(), nil
}

// normalizeID accepts a bare id, a dashed uuid or any Notion URL.
func normalizeID(raw string) string {
	id := strings.TrimSpace(raw)
	if id == "" {
		return ""
	}
	if i := strings.Index(id, "?"); i >= 0 {
		id = id[:i]
	}
	if i := strings.LastIndex(id, "/"); i >= 0 {
		id = id[i+1:]
	}
	if i := strings.LastIndex(id, "-"); i > 0 && len(id)-i-1 == 32 {
		id = id[i+1:]
	}
	return id
}
