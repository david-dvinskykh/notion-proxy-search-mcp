// Package sync keeps the mirror close to Notion: a bootstrap crawl, then
// watermark-driven incremental polling, then a slower reconciliation pass that
// notices deletions. Writes always go to Notion through the official MCP
// server; this package only ever reads.
package sync

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/index"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mirror"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/notion"
)

// Options configures a Syncer.
type Options struct {
	// Slack is subtracted from a watermark before querying. Notion timestamps
	// have minute granularity in filters, so without slack an edit inside the
	// same minute as the watermark can be missed.
	Slack time.Duration
	// MaxBlockDepth bounds recursion into nested blocks.
	MaxBlockDepth int
	// MaxBlocksPerPage caps one page's block tree.
	MaxBlocksPerPage int
	Logf             func(format string, args ...any)
}

// Syncer mirrors a Notion workspace.
type Syncer struct {
	client *notion.Client
	db     *mirror.DB
	idx    *index.Store
	opts   Options
}

// New builds a syncer.
func New(client *notion.Client, db *mirror.DB, idx *index.Store, opts Options) *Syncer {
	if opts.Slack <= 0 {
		opts.Slack = 2 * time.Minute
	}
	if opts.MaxBlockDepth <= 0 {
		opts.MaxBlockDepth = 6
	}
	if opts.MaxBlocksPerPage <= 0 {
		opts.MaxBlocksPerPage = 3000
	}
	if opts.Logf == nil {
		opts.Logf = func(string, ...any) {}
	}
	return &Syncer{client: client, db: db, idx: idx, opts: opts}
}

// Stats summarises one pass.
type Stats struct {
	DataSources int `json:"data_sources"`
	Pages       int `json:"pages_seen"`
	Changed     int `json:"pages_changed"`
	Content     int `json:"content_fetched"`
	Trashed     int `json:"pages_trashed"`
}

// Discover finds every database the integration can see and mirrors its schema.
// Schemas are what the collection:// views are generated from, so this runs
// before any row is stored.
func (s *Syncer) Discover(ctx context.Context) (int, error) {
	var dbIDs []string
	err := s.client.Search(ctx, notion.SearchParams{ObjectType: "data_source"}, func(raw json.RawMessage) bool {
		var ds notion.DataSource
		if err := json.Unmarshal(raw, &ds); err != nil {
			return true
		}
		ds.Raw = raw
		ds.DatabaseParentID = ds.Parent.DatabaseID
		if err := s.db.UpsertDataSource(ctx, &ds); err != nil {
			s.opts.Logf("discover: data source %s: %v", ds.ID, err)
			return true
		}
		if ds.DatabaseParentID != "" {
			dbIDs = append(dbIDs, ds.DatabaseParentID)
		}
		return true
	})
	if err != nil {
		// Older API versions have no data_source object; fall back to searching
		// databases and expanding them.
		s.opts.Logf("discover: data_source search failed (%v), falling back to databases", err)
		if err := s.discoverViaDatabases(ctx); err != nil {
			return 0, err
		}
	}
	for _, id := range uniq(dbIDs) {
		db, err := s.client.RetrieveDatabase(ctx, id)
		if err != nil {
			s.opts.Logf("discover: database %s: %v", id, err)
			continue
		}
		if err := s.db.UpsertDatabase(ctx, db); err != nil {
			s.opts.Logf("discover: store database %s: %v", id, err)
		}
	}
	sources, err := s.db.DataSources(ctx)
	if err != nil {
		return 0, err
	}
	return len(sources), nil
}

func (s *Syncer) discoverViaDatabases(ctx context.Context) error {
	var ids []string
	err := s.client.Search(ctx, notion.SearchParams{ObjectType: "database"}, func(raw json.RawMessage) bool {
		var db notion.Database
		if err := json.Unmarshal(raw, &db); err != nil {
			return true
		}
		db.Raw = raw
		if err := s.db.UpsertDatabase(ctx, &db); err != nil {
			s.opts.Logf("discover: store database %s: %v", db.ID, err)
			return true
		}
		for _, ds := range db.DataSources {
			ids = append(ids, ds.ID)
		}
		if len(db.DataSources) == 0 {
			ids = append(ids, db.ID)
		}
		return true
	})
	if err != nil {
		return err
	}
	for _, id := range uniq(ids) {
		ds, err := s.client.RetrieveDataSource(ctx, id)
		if err != nil {
			s.opts.Logf("discover: data source %s: %v", id, err)
			continue
		}
		if err := s.db.UpsertDataSource(ctx, ds); err != nil {
			s.opts.Logf("discover: store data source %s: %v", id, err)
		}
	}
	return nil
}

// Pass runs one incremental sync. full ignores the watermarks and re-reads
// every row, which is what the bootstrap and the reconciliation pass want.
func (s *Syncer) Pass(ctx context.Context, full bool) (Stats, error) {
	var st Stats
	sources, err := s.db.DataSources(ctx)
	if err != nil {
		return st, err
	}
	st.DataSources = len(sources)

	for _, ds := range sources {
		if err := ctx.Err(); err != nil {
			return st, err
		}
		seen, changed, newWatermark, err := s.syncDataSource(ctx, ds, full)
		if err != nil {
			s.opts.Logf("sync: data source %s (%s): %v", ds.Title, ds.ID, err)
			continue
		}
		st.Pages += seen
		st.Changed += changed
		if newWatermark != "" && newWatermark > ds.Watermark {
			if err := s.db.SetWatermark(ctx, ds.ID, newWatermark); err != nil {
				return st, err
			}
		}
		if full {
			trashed, err := s.reconcileDeletions(ctx, ds.ID)
			if err != nil {
				s.opts.Logf("sync: reconcile %s: %v", ds.ID, err)
			}
			st.Trashed += trashed
		}
	}

	loose, changed, err := s.syncLoosePages(ctx, full)
	if err != nil {
		s.opts.Logf("sync: loose pages: %v", err)
	}
	st.Pages += loose
	st.Changed += changed

	fetched, err := s.FetchPendingContent(ctx, 0)
	if err != nil {
		s.opts.Logf("sync: content: %v", err)
	}
	st.Content = fetched
	return st, nil
}

// syncDataSource pulls rows edited since the watermark.
func (s *Syncer) syncDataSource(ctx context.Context, ds mirror.DataSourceRow, full bool) (seen, changed int, newWatermark string, err error) {
	params := notion.QueryParams{Sorts: notion.AscendingEditedSort()}
	if !full && ds.Watermark != "" {
		since := ds.Watermark
		if t, parseErr := time.Parse(time.RFC3339, ds.Watermark); parseErr == nil {
			since = t.Add(-s.opts.Slack).UTC().Format(time.RFC3339)
		}
		params.Filter = notion.EditedAfterFilter(since)
	}

	var loopErr error
	err = s.client.QueryDataSource(ctx, ds.ID, params, func(raw json.RawMessage) bool {
		var page notion.Page
		if loopErr = json.Unmarshal(raw, &page); loopErr != nil {
			return false
		}
		seen++
		var didChange bool
		didChange, loopErr = s.db.UpsertPage(ctx, &page, ds.ID)
		if loopErr != nil {
			return false
		}
		if didChange {
			changed++
		}
		if page.LastEditedTime > newWatermark {
			newWatermark = page.LastEditedTime
		}
		return true
	})
	if err == nil {
		err = loopErr
	}
	return seen, changed, newWatermark, err
}

// syncLoosePages mirrors pages that live outside a database. Search is eventually
// consistent, so it is used only for these; database rows come from the strongly
// consistent query endpoint.
func (s *Syncer) syncLoosePages(ctx context.Context, full bool) (seen, changed int, err error) {
	watermark, err := s.db.Meta(ctx, "loose_watermark")
	if err != nil {
		return 0, 0, err
	}
	cutoff := ""
	if !full && watermark != "" {
		if t, parseErr := time.Parse(time.RFC3339, watermark); parseErr == nil {
			cutoff = t.Add(-s.opts.Slack).UTC().Format(time.RFC3339)
		}
	}

	newest := watermark
	var loopErr error
	err = s.client.Search(ctx, notion.SearchParams{ObjectType: "page"}, func(raw json.RawMessage) bool {
		var page notion.Page
		if loopErr = json.Unmarshal(raw, &page); loopErr != nil {
			return false
		}
		if cutoff != "" && page.LastEditedTime < cutoff {
			// Search returns newest first, so the first older page ends the walk.
			return false
		}
		if page.LastEditedTime > newest {
			newest = page.LastEditedTime
		}
		// Rows of a data source are handled by the query endpoint.
		if page.Parent.Type == "data_source" || page.Parent.Type == "database_id" {
			return true
		}
		seen++
		var didChange bool
		didChange, loopErr = s.db.UpsertPage(ctx, &page, "")
		if loopErr != nil {
			return false
		}
		if didChange {
			changed++
		}
		return true
	})
	if err == nil {
		err = loopErr
	}
	if err != nil {
		return seen, changed, err
	}
	if newest != watermark {
		if err := s.db.SetMeta(ctx, "loose_watermark", newest); err != nil {
			return seen, changed, err
		}
	}
	return seen, changed, nil
}

// reconcileDeletions marks pages that Notion no longer returns as trashed. A
// deleted page never shows up in an incremental query, so only a full listing
// can notice it.
func (s *Syncer) reconcileDeletions(ctx context.Context, dataSourceID string) (int, error) {
	known, err := s.db.PageIDs(ctx, dataSourceID)
	if err != nil {
		return 0, err
	}
	live := map[string]bool{}
	var loopErr error
	err = s.client.QueryDataSource(ctx, dataSourceID, notion.QueryParams{}, func(raw json.RawMessage) bool {
		var page struct {
			ID string `json:"id"`
		}
		if loopErr = json.Unmarshal(raw, &page); loopErr != nil {
			return false
		}
		live[page.ID] = true
		return true
	})
	if err == nil {
		err = loopErr
	}
	if err != nil {
		return 0, err
	}
	trashed := 0
	for id := range known {
		if live[id] {
			continue
		}
		if err := s.db.MarkTrashed(ctx, id); err != nil {
			return trashed, err
		}
		if s.idx != nil {
			if err := s.idx.DropPage(ctx, id); err != nil {
				return trashed, err
			}
		}
		trashed++
	}
	return trashed, nil
}

// FetchPendingContent downloads block trees for pages whose body is missing or
// out of date. limit of 0 means every pending page.
func (s *Syncer) FetchPendingContent(ctx context.Context, limit int) (int, error) {
	query := `
		SELECT id, last_edited_time FROM pages
		WHERE in_trash = 0 AND (content_synced = '' OR content_synced <> last_edited_time)
		ORDER BY last_edited_time DESC`
	args := []any{}
	if limit > 0 {
		query += ` LIMIT ?`
		args = append(args, limit)
	}
	rows, err := s.db.SQL().QueryContext(ctx, query, args...)
	if err != nil {
		return 0, err
	}
	var ids []string
	for rows.Next() {
		var id, edited string
		if err := rows.Scan(&id, &edited); err != nil {
			rows.Close()
			return 0, err
		}
		ids = append(ids, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return 0, err
	}

	done := 0
	for _, id := range ids {
		if err := ctx.Err(); err != nil {
			return done, err
		}
		if err := s.FetchContent(ctx, id); err != nil {
			s.opts.Logf("sync: content for %s: %v", id, err)
			continue
		}
		done++
	}
	return done, nil
}

// FetchContent downloads and renders one page's block tree.
func (s *Syncer) FetchContent(ctx context.Context, pageID string) error {
	budget := s.opts.MaxBlocksPerPage
	nodes, err := s.fetchChildren(ctx, pageID, s.opts.MaxBlockDepth, &budget)
	if err != nil {
		return err
	}
	markdown := mirror.RenderMarkdown(nodes)
	if err := s.db.SetContent(ctx, pageID, markdown); err != nil {
		return err
	}
	if s.idx != nil {
		return s.idx.Reindex(ctx, pageID)
	}
	return nil
}

func (s *Syncer) fetchChildren(ctx context.Context, id string, depth int, budget *int) ([]mirror.Node, error) {
	if depth <= 0 || *budget <= 0 {
		return nil, nil
	}
	blocks, err := s.client.BlockChildren(ctx, id)
	if err != nil {
		var apiErr *notion.APIError
		// A page the integration cannot open is not a sync failure.
		if asAPIError(err, &apiErr) && (apiErr.Status == 404 || apiErr.Status == 403) {
			return nil, nil
		}
		return nil, err
	}
	out := make([]mirror.Node, 0, len(blocks))
	for _, b := range blocks {
		*budget--
		if *budget <= 0 {
			break
		}
		node := mirror.Node{Block: b}
		// child_page blocks are separate pages with their own mirror rows;
		// recursing into them would duplicate their text.
		if b.HasChildren && b.Type != "child_page" && b.Type != "child_database" {
			children, err := s.fetchChildren(ctx, b.ID, depth-1, budget)
			if err != nil {
				return nil, err
			}
			node.Children = children
		}
		out = append(out, node)
	}
	return out, nil
}

// SyncUsers mirrors the workspace member list so people properties can be
// rendered as names.
func (s *Syncer) SyncUsers(ctx context.Context) error {
	users, err := s.client.Users(ctx)
	if err != nil {
		return err
	}
	return s.db.UpsertUsers(ctx, users)
}

// ResyncPage re-reads one page from Notion, for the tool that wants the mirror
// current right now rather than at the next poll.
func (s *Syncer) ResyncPage(ctx context.Context, pageID string) error {
	page, err := s.client.RetrievePage(ctx, pageID)
	if err != nil {
		return err
	}
	dataSourceID := ""
	if page.Parent.Type == "data_source" {
		dataSourceID = page.Parent.DataSourceID
	} else if page.Parent.DatabaseID != "" {
		dataSourceID = page.Parent.DatabaseID
	}
	if _, err := s.db.UpsertPage(ctx, page, dataSourceID); err != nil {
		return err
	}
	return s.FetchContent(ctx, page.ID)
}

func uniq(values []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(values))
	for _, v := range values {
		if v == "" || seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}
	return out
}

// asAPIError is errors.As without importing errors in the hot path of a loop
// that already carries its own error handling.
func asAPIError(err error, target **notion.APIError) bool {
	for err != nil {
		if e, ok := err.(*notion.APIError); ok {
			*target = e
			return true
		}
		unwrapper, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = unwrapper.Unwrap()
	}
	return false
}

// Describe renders a short status line for the sync log.
func (st Stats) Describe() string {
	parts := []string{
		fmt.Sprintf("sources=%d", st.DataSources),
		fmt.Sprintf("pages=%d", st.Pages),
		fmt.Sprintf("changed=%d", st.Changed),
		fmt.Sprintf("content=%d", st.Content),
	}
	if st.Trashed > 0 {
		parts = append(parts, fmt.Sprintf("trashed=%d", st.Trashed))
	}
	return strings.Join(parts, " ")
}
