package mirror

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/notion"
)

// DataSourceRow is a mirrored data source.
type DataSourceRow struct {
	ID         string
	DatabaseID string
	Title      string
	SchemaJSON string
	Watermark  string
}

// PageRow is a mirrored page, with Markdown when the content has been fetched.
type PageRow struct {
	ID             string
	Object         string
	DataSourceID   string
	ParentID       string
	ParentType     string
	Title          string
	URL            string
	Icon           string
	CreatedTime    string
	LastEditedTime string
	Archived       bool
	InTrash        bool
	Properties     string
	Markdown       string
	ContentSynced  string
}

// UpsertDatabase stores a database container.
func (d *DB) UpsertDatabase(ctx context.Context, db *notion.Database) error {
	_, err := d.sql.ExecContext(ctx, `
		INSERT INTO databases(id,title,url,parent_id,in_trash,raw) VALUES(?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET title=excluded.title, url=excluded.url,
		  parent_id=excluded.parent_id, in_trash=excluded.in_trash, raw=excluded.raw`,
		db.ID, notion.PlainTextOf(db.Title), db.URL, db.Parent.ParentID(), boolInt(db.InTrash || db.Archived), string(db.Raw))
	return err
}

// UpsertDataSource stores a data source and rebuilds its SQL view when the
// schema changed. The view is what makes mirrored SQL a drop-in replacement.
func (d *DB) UpsertDataSource(ctx context.Context, ds *notion.DataSource) error {
	schema := string(ds.Schema)
	if schema == "" {
		schema = "{}"
	}
	var prevSchema string
	err := d.sql.QueryRowContext(ctx, `SELECT schema_json FROM data_sources WHERE id=?`, ds.ID).Scan(&prevSchema)
	if err != nil && err != sql.ErrNoRows {
		return err
	}

	viewSQL, err := ViewSQL(ds.ID, schema)
	if err != nil {
		return fmt.Errorf("build view for %s: %w", ds.ID, err)
	}
	if prevSchema != schema {
		if _, err := d.sql.ExecContext(ctx, `DROP VIEW IF EXISTS `+quoteIdent(ViewName(ds.ID))); err != nil {
			return err
		}
		if _, err := d.sql.ExecContext(ctx, viewSQL); err != nil {
			return fmt.Errorf("create view for %s: %w", ds.ID, err)
		}
	}

	_, err = d.sql.ExecContext(ctx, `
		INSERT INTO data_sources(id,database_id,title,schema_json,view_sql,in_trash)
		VALUES(?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET database_id=excluded.database_id, title=excluded.title,
		  schema_json=excluded.schema_json, view_sql=excluded.view_sql, in_trash=excluded.in_trash`,
		ds.ID, ds.DatabaseParentID, notion.PlainTextOf(ds.Title), schema, viewSQL, boolInt(ds.InTrash || ds.Archived))
	return err
}

// EnsureViews recreates every data source view, used after a fresh open.
func (d *DB) EnsureViews(ctx context.Context) error {
	rows, err := d.sql.QueryContext(ctx, `SELECT id, view_sql FROM data_sources WHERE view_sql <> ''`)
	if err != nil {
		return err
	}
	defer rows.Close()
	type pair struct{ id, sqlText string }
	var pending []pair
	for rows.Next() {
		var p pair
		if err := rows.Scan(&p.id, &p.sqlText); err != nil {
			return err
		}
		pending = append(pending, p)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	for _, p := range pending {
		if _, err := d.sql.ExecContext(ctx, `DROP VIEW IF EXISTS `+quoteIdent(ViewName(p.id))); err != nil {
			return err
		}
		if _, err := d.sql.ExecContext(ctx, p.sqlText); err != nil {
			return fmt.Errorf("recreate view %s: %w", p.id, err)
		}
	}
	return nil
}

// DataSources lists mirrored data sources.
func (d *DB) DataSources(ctx context.Context) ([]DataSourceRow, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT id,database_id,title,schema_json,watermark FROM data_sources WHERE in_trash=0 ORDER BY title`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []DataSourceRow
	for rows.Next() {
		var r DataSourceRow
		if err := rows.Scan(&r.ID, &r.DatabaseID, &r.Title, &r.SchemaJSON, &r.Watermark); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// SetWatermark records the newest last_edited_time synced for a data source.
func (d *DB) SetWatermark(ctx context.Context, dataSourceID, watermark string) error {
	_, err := d.sql.ExecContext(ctx, `UPDATE data_sources SET watermark=? WHERE id=?`, watermark, dataSourceID)
	return err
}

// UpsertPage stores page metadata and its flattened properties. It reports
// whether the page is new or its last edit moved, which tells the caller
// whether the block content has to be refetched.
func (d *DB) UpsertPage(ctx context.Context, p *notion.Page, dataSourceID string) (changed bool, err error) {
	cols, rels, err := FlattenProperties(p.Properties)
	if err != nil {
		return false, fmt.Errorf("page %s: %w", p.ID, err)
	}
	title := titleFromColumns(p.Properties, cols)

	tx, err := d.sql.BeginTx(ctx, nil)
	if err != nil {
		return false, err
	}
	defer tx.Rollback()

	var prevEdited string
	err = tx.QueryRowContext(ctx, `SELECT last_edited_time FROM pages WHERE id=?`, p.ID).Scan(&prevEdited)
	switch {
	case err == sql.ErrNoRows:
		changed = true
	case err != nil:
		return false, err
	default:
		changed = prevEdited != p.LastEditedTime
	}

	_, err = tx.ExecContext(ctx, `
		INSERT INTO pages(id,object,data_source_id,parent_id,parent_type,title,url,icon,
		                  created_time,last_edited_time,archived,in_trash,properties,synced_at)
		VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(id) DO UPDATE SET object=excluded.object,
		  -- A page can arrive from the search endpoint with no data source
		  -- attached. Keeping the known one matters: every collection:// view
		  -- joins on it, so clearing it empties the view for that row.
		  data_source_id=CASE WHEN excluded.data_source_id <> '' THEN excluded.data_source_id ELSE pages.data_source_id END,
		  parent_id=excluded.parent_id, parent_type=excluded.parent_type, title=excluded.title,
		  url=excluded.url, icon=excluded.icon, created_time=excluded.created_time,
		  last_edited_time=excluded.last_edited_time, archived=excluded.archived,
		  in_trash=excluded.in_trash, properties=excluded.properties, synced_at=excluded.synced_at`,
		p.ID, orDefault(p.Object.Object, "page"), dataSourceID, p.Parent.ParentID(), p.Parent.Type,
		title, p.URL, iconOf(p.Icon), p.CreatedTime, p.LastEditedTime,
		boolInt(p.Archived), boolInt(p.InTrash), string(p.Properties), nowISO())
	if err != nil {
		return false, err
	}

	if _, err := tx.ExecContext(ctx, `DELETE FROM page_props WHERE page_id=?`, p.ID); err != nil {
		return false, err
	}
	insertProp, err := tx.PrepareContext(ctx, `INSERT INTO page_props(page_id,col,value) VALUES(?,?,?)`)
	if err != nil {
		return false, err
	}
	defer insertProp.Close()
	for _, c := range cols {
		if _, err := insertProp.ExecContext(ctx, p.ID, c.Name, c.Value); err != nil {
			return false, err
		}
	}

	if _, err := tx.ExecContext(ctx, `DELETE FROM relations WHERE from_id=?`, p.ID); err != nil {
		return false, err
	}
	insertRel, err := tx.PrepareContext(ctx,
		`INSERT OR IGNORE INTO relations(from_id,prop,to_id) VALUES(?,?,?)`)
	if err != nil {
		return false, err
	}
	defer insertRel.Close()
	for _, r := range rels {
		if _, err := insertRel.ExecContext(ctx, p.ID, r.Prop, r.To); err != nil {
			return false, err
		}
	}
	return changed, tx.Commit()
}

// SetContent stores rendered Markdown for a page. content_synced records the
// page's last_edited_time as of the fetch, not the wall clock: staleness is
// then "Notion says this page changed since we read its blocks", which is the
// actual question, and it survives clock skew between the board and Notion.
func (d *DB) SetContent(ctx context.Context, pageID, markdown string) error {
	hash := ContentHash(markdown)
	_, err := d.sql.ExecContext(ctx,
		`UPDATE pages SET content_markdown=?, content_hash=?, content_runes=?,
		       content_synced=last_edited_time WHERE id=?`,
		markdown, hash, len([]rune(markdown)), pageID)
	return err
}

// MarkTrashed flags a page as gone without deleting its history.
func (d *DB) MarkTrashed(ctx context.Context, pageID string) error {
	_, err := d.sql.ExecContext(ctx, `UPDATE pages SET in_trash=1, synced_at=? WHERE id=?`, nowISO(), pageID)
	return err
}

// Page loads one mirrored page by id.
func (d *DB) Page(ctx context.Context, id string) (*PageRow, error) {
	row := d.sql.QueryRowContext(ctx, `
		SELECT id,object,data_source_id,parent_id,parent_type,title,url,icon,created_time,
		       last_edited_time,archived,in_trash,properties,content_markdown,content_synced
		FROM pages WHERE id=?`, id)
	var p PageRow
	var archived, inTrash int
	if err := row.Scan(&p.ID, &p.Object, &p.DataSourceID, &p.ParentID, &p.ParentType, &p.Title, &p.URL,
		&p.Icon, &p.CreatedTime, &p.LastEditedTime, &archived, &inTrash, &p.Properties,
		&p.Markdown, &p.ContentSynced); err != nil {
		return nil, err
	}
	p.Archived, p.InTrash = archived != 0, inTrash != 0
	return &p, nil
}

// PageIDs returns every non-trashed page id in a data source, for the
// reconciliation pass that notices deletions.
func (d *DB) PageIDs(ctx context.Context, dataSourceID string) (map[string]bool, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT id FROM pages WHERE data_source_id=? AND in_trash=0`, dataSourceID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]bool{}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out[id] = true
	}
	return out, rows.Err()
}

// UpsertUsers stores the workspace member list.
func (d *DB) UpsertUsers(ctx context.Context, users []notion.User) error {
	tx, err := d.sql.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	for _, u := range users {
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO users(id,name,email) VALUES(?,?,?)
			ON CONFLICT(id) DO UPDATE SET name=excluded.name, email=excluded.email`,
			u.ID, u.Name, u.Person.Email); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// Counts reports mirror size for the status tool.
func (d *DB) Counts(ctx context.Context) (map[string]int, error) {
	out := map[string]int{}
	for name, query := range map[string]string{
		"databases":    `SELECT COUNT(*) FROM databases WHERE in_trash=0`,
		"data_sources": `SELECT COUNT(*) FROM data_sources WHERE in_trash=0`,
		"pages":        `SELECT COUNT(*) FROM pages WHERE in_trash=0`,
		"with_content": `SELECT COUNT(*) FROM pages WHERE in_trash=0 AND content_markdown<>''`,
		"chunks":       `SELECT COUNT(*) FROM chunks`,
		"vectors":      `SELECT COUNT(*) FROM vectors`,
	} {
		var n int
		if err := d.sql.QueryRowContext(ctx, query).Scan(&n); err != nil {
			return nil, err
		}
		out[name] = n
	}
	return out, nil
}

// RecentLog returns the last sync log lines.
func (d *DB) RecentLog(ctx context.Context, limit int) ([]string, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT at || ' [' || kind || '] ' || detail FROM sync_log ORDER BY rowid DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var s string
		if err := rows.Scan(&s); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

// ContentHash is the change detector used for reindexing decisions.
func ContentHash(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:8])
}

func titleFromColumns(props json.RawMessage, cols []Column) string {
	// The title property is the one Notion types as "title"; find its name
	// once rather than trusting a conventional column name.
	var raw map[string]struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(props, &raw); err == nil {
		for name, meta := range raw {
			if meta.Type != "title" {
				continue
			}
			want := ColumnName(name)
			for _, c := range cols {
				if c.Name == want {
					if s, ok := c.Value.(string); ok {
						return s
					}
				}
			}
		}
	}
	return ""
}

func iconOf(icon json.RawMessage) string {
	if len(icon) == 0 {
		return ""
	}
	var v struct {
		Type  string `json:"type"`
		Emoji string `json:"emoji"`
	}
	if err := json.Unmarshal(icon, &v); err != nil {
		return ""
	}
	if v.Emoji != "" {
		return v.Emoji
	}
	return ""
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func orDefault(s, fallback string) string {
	if strings.TrimSpace(s) == "" {
		return fallback
	}
	return s
}

func nowISO() string { return time.Now().UTC().Format(time.RFC3339) }
