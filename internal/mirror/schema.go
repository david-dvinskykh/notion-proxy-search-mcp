// Package mirror keeps a local SQLite copy of the Notion pages an integration
// can see: properties flattened into queryable columns, block trees rendered
// to Markdown, and one SQL view per data source so queries written against the
// official Notion MCP server run unchanged against the mirror.
package mirror

import (
	"context"
	"database/sql"
	"fmt"

	_ "modernc.org/sqlite" // pure-Go driver: the binary cross-compiles without cgo
)

// schemaVersion is bumped whenever the DDL below changes in a way that needs a
// rebuild. The daemon drops and re-crawls when it finds an older mirror.
const schemaVersion = 2

const ddl = `
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
) WITHOUT ROWID;

-- Databases are containers; data sources hold the schema and the rows.
CREATE TABLE IF NOT EXISTS databases (
  id          TEXT PRIMARY KEY,
  title       TEXT NOT NULL DEFAULT '',
  url         TEXT NOT NULL DEFAULT '',
  parent_id   TEXT NOT NULL DEFAULT '',
  in_trash    INTEGER NOT NULL DEFAULT 0,
  raw         TEXT NOT NULL DEFAULT '{}'
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS data_sources (
  id           TEXT PRIMARY KEY,
  database_id  TEXT NOT NULL DEFAULT '',
  title        TEXT NOT NULL DEFAULT '',
  schema_json  TEXT NOT NULL DEFAULT '{}',
  view_sql     TEXT NOT NULL DEFAULT '',
  watermark    TEXT NOT NULL DEFAULT '',
  in_trash     INTEGER NOT NULL DEFAULT 0
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS pages (
  id               TEXT PRIMARY KEY,
  object           TEXT NOT NULL DEFAULT 'page',
  data_source_id   TEXT NOT NULL DEFAULT '',
  parent_id        TEXT NOT NULL DEFAULT '',
  parent_type      TEXT NOT NULL DEFAULT '',
  title            TEXT NOT NULL DEFAULT '',
  url              TEXT NOT NULL DEFAULT '',
  icon             TEXT NOT NULL DEFAULT '',
  created_time     TEXT NOT NULL DEFAULT '',
  last_edited_time TEXT NOT NULL DEFAULT '',
  archived         INTEGER NOT NULL DEFAULT 0,
  in_trash         INTEGER NOT NULL DEFAULT 0,
  properties       TEXT NOT NULL DEFAULT '{}',
  content_markdown TEXT NOT NULL DEFAULT '',
  content_runes    INTEGER NOT NULL DEFAULT 0,
  content_synced   TEXT NOT NULL DEFAULT '',
  content_hash     TEXT NOT NULL DEFAULT '',
  indexed_hash     TEXT NOT NULL DEFAULT '',
  synced_at        TEXT NOT NULL DEFAULT ''
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS pages_by_source ON pages(data_source_id, last_edited_time);
CREATE INDEX IF NOT EXISTS pages_by_parent ON pages(parent_id);
CREATE INDEX IF NOT EXISTS pages_stale ON pages(indexed_hash, content_hash);
-- The embedding queue walks pages cheapest-first, so the cost column is indexed.
CREATE INDEX IF NOT EXISTS pages_by_cost ON pages(content_runes);

-- Flattened property values. One row per (page, column); the column name is
-- already the name a SQL query uses, including the "date:Name:start" form.
CREATE TABLE IF NOT EXISTS page_props (
  page_id TEXT NOT NULL,
  col     TEXT NOT NULL,
  value, -- no affinity: numbers stay numeric, everything else is text
  PRIMARY KEY (page_id, col)
) WITHOUT ROWID;

-- Relation properties, denormalised so the retriever can walk the graph.
CREATE TABLE IF NOT EXISTS relations (
  from_id  TEXT NOT NULL,
  prop     TEXT NOT NULL,
  to_id    TEXT NOT NULL,
  PRIMARY KEY (from_id, prop, to_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS relations_reverse ON relations(to_id);

CREATE TABLE IF NOT EXISTS users (
  id    TEXT PRIMARY KEY,
  name  TEXT NOT NULL DEFAULT '',
  email TEXT NOT NULL DEFAULT ''
) WITHOUT ROWID;

-- Retrieval units. A short database row is one chunk; a long page is split on
-- heading boundaries with the heading path kept for context.
CREATE TABLE IF NOT EXISTS chunks (
  id        INTEGER PRIMARY KEY,
  page_id   TEXT NOT NULL,
  ord       INTEGER NOT NULL,
  heading   TEXT NOT NULL DEFAULT '',
  text      TEXT NOT NULL,
  hash      TEXT NOT NULL DEFAULT ''
);

-- Not unique: a reindex renumbers ord while the previous numbering is still in
-- place, and a transient collision is not a data error.
CREATE INDEX IF NOT EXISTS chunks_by_page ON chunks(page_id, ord);

CREATE TABLE IF NOT EXISTS vectors (
  chunk_id INTEGER PRIMARY KEY,
  dim      INTEGER NOT NULL,
  scale    REAL NOT NULL,
  vec      BLOB NOT NULL,
  model    TEXT NOT NULL DEFAULT ''
) WITHOUT ROWID;

-- A plain (not contentless) FTS5 table: it duplicates the chunk text, which
-- costs disk the Pi has, and in exchange a delete is an ordinary DELETE rather
-- than the contentless 'delete' incantation that needs the original values.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text,
  heading,
  title,
  tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS sync_log (
  at      TEXT NOT NULL,
  kind    TEXT NOT NULL,
  detail  TEXT NOT NULL DEFAULT ''
);
`

// Open opens or creates the mirror database at path.
func Open(ctx context.Context, path string) (*DB, error) {
	// _time_format keeps timestamps as the ISO strings Notion returns.
	dsn := path + "?_pragma=busy_timeout(10000)&_pragma=journal_mode(WAL)&_time_format=sqlite"
	sqlDB, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	// A single writer avoids SQLITE_BUSY churn; readers go through the same
	// pool because WAL keeps them from blocking each other anyway.
	sqlDB.SetMaxOpenConns(4)
	if _, err := sqlDB.ExecContext(ctx, ddl); err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("mirror: apply schema: %w", err)
	}
	db := &DB{sql: sqlDB}
	if err := db.checkVersion(ctx); err != nil {
		sqlDB.Close()
		return nil, err
	}
	return db, nil
}

// OpenReadOnly opens an existing mirror without applying DDL, for the stdio
// process when it cannot reach the daemon.
func OpenReadOnly(ctx context.Context, path string) (*DB, error) {
	dsn := "file:" + path + "?mode=ro&_pragma=busy_timeout(5000)&_time_format=sqlite"
	sqlDB, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	if err := sqlDB.PingContext(ctx); err != nil {
		sqlDB.Close()
		return nil, err
	}
	return &DB{sql: sqlDB, readOnly: true}, nil
}

// DB is the mirror handle.
type DB struct {
	sql      *sql.DB
	readOnly bool
}

// SQL exposes the underlying handle for the query tool and the index.
func (d *DB) SQL() *sql.DB { return d.sql }

// ReadOnly reports whether writes are possible.
func (d *DB) ReadOnly() bool { return d.readOnly }

// Close releases the database.
func (d *DB) Close() error { return d.sql.Close() }

func (d *DB) checkVersion(ctx context.Context) error {
	var got string
	err := d.sql.QueryRowContext(ctx, `SELECT value FROM meta WHERE key='schema_version'`).Scan(&got)
	switch {
	case err == sql.ErrNoRows:
		_, err = d.sql.ExecContext(ctx,
			`INSERT INTO meta(key,value) VALUES('schema_version',?)`, fmt.Sprint(schemaVersion))
		return err
	case err != nil:
		return err
	case got != fmt.Sprint(schemaVersion):
		return fmt.Errorf("mirror: schema version %s on disk, binary expects %d: delete the mirror to rebuild", got, schemaVersion)
	}
	return nil
}

// Meta reads a bookkeeping value.
func (d *DB) Meta(ctx context.Context, key string) (string, error) {
	var v string
	err := d.sql.QueryRowContext(ctx, `SELECT value FROM meta WHERE key=?`, key).Scan(&v)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return v, err
}

// SetMeta stores a bookkeeping value.
func (d *DB) SetMeta(ctx context.Context, key, value string) error {
	_, err := d.sql.ExecContext(ctx,
		`INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
		key, value)
	return err
}

// Logf appends a line to the sync log, which the status tool reports.
func (d *DB) Logf(ctx context.Context, at, kind, format string, args ...any) {
	_, _ = d.sql.ExecContext(ctx, `INSERT INTO sync_log(at,kind,detail) VALUES(?,?,?)`,
		at, kind, fmt.Sprintf(format, args...))
	// Keep the log small: this runs on a Raspberry Pi, not a log server.
	_, _ = d.sql.ExecContext(ctx,
		`DELETE FROM sync_log WHERE rowid NOT IN (SELECT rowid FROM sync_log ORDER BY rowid DESC LIMIT 500)`)
}
