package index

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"sync"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/embed"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mirror"
)

// Embedder is the part of the embedding model the index needs. Keeping it an
// interface lets the index be tested without loading weights, and lets the
// stdio process run keyword-only when no model is installed.
type Embedder interface {
	EmbedQuery(text string) []float32
	EmbedPassage(text string) []float32
	Dim() int
	Name() string
}

// vector is one indexed chunk embedding plus the metadata a filter needs, kept
// in memory so a query never reads 30k rows back out of SQLite.
type vector struct {
	chunkID    int64
	pageID     string
	sourceID   string
	lastEdited string
	scale      float32
	data       []int8
}

// Store owns the retrieval side of the mirror.
type Store struct {
	db  *mirror.DB
	emb Embedder

	mu   sync.RWMutex
	vecs []vector
	pos  map[int64]int
	skip []string
}

// New builds a store. emb may be nil, in which case only keyword search works.
func New(db *mirror.DB, emb Embedder) *Store {
	return &Store{db: db, emb: emb, pos: map[int64]int{}}
}

// HasEmbedder reports whether vector search is available.
func (s *Store) HasEmbedder() bool { return s.emb != nil }

// SkipSources excludes data sources from the vector index. Keyword search still
// covers them, and they are still mirrored and queryable by SQL.
//
// This exists because of a measurement. Workspaces routinely hold one
// append-only, journal-shaped database whose pages dwarf everything else — daily
// log entries of a hundred thousand characters each, six times more text than
// every other database combined — and those pages are the least worth embedding,
// because a chronological log is retrieved by date and by literal string rather
// than by paraphrase. Excluding one such data source turned a measured 15-hour
// backfill into a 2-hour one without losing a single answer an embedding would
// have given.
func (s *Store) SkipSources(ids []string) {
	s.mu.Lock()
	s.skip = append([]string(nil), ids...)
	s.mu.Unlock()
}

// skippedSources returns the current exclusion list.
func (s *Store) skippedSources() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]string(nil), s.skip...)
}

// Skipped reports which data sources are excluded from the vector index.
func (s *Store) Skipped() []string { return s.skippedSources() }

// ModelName is the identifier stored with every vector.
func (s *Store) ModelName() string {
	if s.emb == nil {
		return ""
	}
	return s.emb.Name()
}

// LoadVectors fills the in-memory vector cache from the mirror.
func (s *Store) LoadVectors(ctx context.Context) error {
	rows, err := s.db.SQL().QueryContext(ctx, `
		SELECT v.chunk_id, c.page_id, p.data_source_id, p.last_edited_time, v.scale, v.vec
		FROM vectors v
		JOIN chunks c ON c.id = v.chunk_id
		JOIN pages p ON p.id = c.page_id
		WHERE p.in_trash = 0`)
	if err != nil {
		return err
	}
	defer rows.Close()

	var loaded []vector
	pos := map[int64]int{}
	for rows.Next() {
		var v vector
		var blob []byte
		if err := rows.Scan(&v.chunkID, &v.pageID, &v.sourceID, &v.lastEdited, &v.scale, &blob); err != nil {
			return err
		}
		v.data = make([]int8, len(blob))
		for i, b := range blob {
			v.data[i] = int8(b)
		}
		pos[v.chunkID] = len(loaded)
		loaded = append(loaded, v)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	s.vecs, s.pos = loaded, pos
	s.mu.Unlock()
	return nil
}

// VectorCount reports how many embeddings are cached.
func (s *Store) VectorCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.vecs)
}

// Reindex rebuilds the chunks of one page. Chunks whose text is unchanged keep
// their id and their embedding, so editing one section of a long page does not
// force the whole page through the model again.
func (s *Store) Reindex(ctx context.Context, pageID string) error {
	page, err := s.db.Page(ctx, pageID)
	if err == sql.ErrNoRows {
		return s.DropPage(ctx, pageID)
	}
	if err != nil {
		return err
	}
	if page.InTrash {
		return s.DropPage(ctx, pageID)
	}

	digest, err := PropertyDigest(ctx, s.db.SQL(), pageID, page.Title)
	if err != nil {
		return err
	}
	fresh := BuildChunks(page, digest)

	tx, err := s.db.SQL().BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	reusable, err := existingChunks(ctx, tx, pageID)
	if err != nil {
		return err
	}

	var dropped []int64
	for ord := range fresh {
		c := &fresh[ord]
		c.Ord = ord
		hash := mirror.ContentHash(c.Heading + "\x00" + c.Text)
		if ids := reusable[hash]; len(ids) > 0 {
			c.ID = ids[0]
			reusable[hash] = ids[1:]
			if _, err := tx.ExecContext(ctx, `UPDATE chunks SET ord=? WHERE id=?`, ord, c.ID); err != nil {
				return err
			}
			continue
		}
		res, err := tx.ExecContext(ctx,
			`INSERT INTO chunks(page_id, ord, heading, text, hash) VALUES(?,?,?,?,?)`,
			pageID, ord, c.Heading, c.Text, hash)
		if err != nil {
			return err
		}
		c.ID, err = res.LastInsertId()
		if err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO chunks_fts(rowid, text, heading, title) VALUES(?,?,?,?)`,
			c.ID, c.Text, c.Heading, page.Title); err != nil {
			return err
		}
	}
	for _, ids := range reusable {
		dropped = append(dropped, ids...)
	}
	for _, id := range dropped {
		if err := deleteChunk(ctx, tx, id); err != nil {
			return err
		}
	}
	if _, err := tx.ExecContext(ctx,
		`UPDATE pages SET indexed_hash=content_hash WHERE id=?`, pageID); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return err
	}

	if len(dropped) > 0 {
		s.mu.Lock()
		s.forgetLocked(dropped)
		s.mu.Unlock()
	}
	return nil
}

// ord numbers are reassigned on every reindex, so a chunk's identity is its
// text; existingChunks groups the current rows by content hash.
func existingChunks(ctx context.Context, tx *sql.Tx, pageID string) (map[string][]int64, error) {
	rows, err := tx.QueryContext(ctx, `SELECT id, hash FROM chunks WHERE page_id=? ORDER BY ord`, pageID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string][]int64{}
	for rows.Next() {
		var id int64
		var hash string
		if err := rows.Scan(&id, &hash); err != nil {
			return nil, err
		}
		out[hash] = append(out[hash], id)
	}
	return out, rows.Err()
}

func deleteChunk(ctx context.Context, tx *sql.Tx, id int64) error {
	for _, stmt := range []string{
		`DELETE FROM chunks_fts WHERE rowid=?`,
		`DELETE FROM vectors WHERE chunk_id=?`,
		`DELETE FROM chunks WHERE id=?`,
	} {
		if _, err := tx.ExecContext(ctx, stmt, id); err != nil {
			return err
		}
	}
	return nil
}

// DropPage removes every chunk of a page from the index.
func (s *Store) DropPage(ctx context.Context, pageID string) error {
	tx, err := s.db.SQL().BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	ids, err := chunkIDs(ctx, tx, pageID)
	if err != nil {
		return err
	}
	for _, id := range ids {
		if err := deleteChunk(ctx, tx, id); err != nil {
			return err
		}
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	s.mu.Lock()
	s.forgetLocked(ids)
	s.mu.Unlock()
	return nil
}

func chunkIDs(ctx context.Context, tx *sql.Tx, pageID string) ([]int64, error) {
	rows, err := tx.QueryContext(ctx, `SELECT id FROM chunks WHERE page_id=?`, pageID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out = append(out, id)
	}
	return out, rows.Err()
}

// forgetLocked removes vectors from the cache. Callers hold the write lock.
func (s *Store) forgetLocked(ids []int64) {
	drop := make(map[int64]bool, len(ids))
	for _, id := range ids {
		drop[id] = true
	}
	kept := s.vecs[:0]
	pos := make(map[int64]int, len(s.vecs))
	for _, v := range s.vecs {
		if drop[v.chunkID] {
			continue
		}
		pos[v.chunkID] = len(kept)
		kept = append(kept, v)
	}
	s.vecs, s.pos = kept, pos
}

// PendingCount reports how many chunks still need an embedding for the current
// model.
func (s *Store) PendingCount(ctx context.Context) (int, error) {
	if s.emb == nil {
		return 0, nil
	}
	where, args := s.pendingFilter()
	var n int
	err := s.db.SQL().QueryRowContext(ctx, `
		SELECT COUNT(*) FROM chunks c
		JOIN pages p ON p.id = c.page_id AND p.in_trash = 0
		LEFT JOIN vectors v ON v.chunk_id = c.id
		WHERE (v.chunk_id IS NULL OR v.model <> ?)`+where, args...).Scan(&n)
	return n, err
}

// EmbedPending embeds up to limit chunks and returns how many it did. The
// daemon calls this in a slow loop: on four Cortex-A72 cores an embedding costs
// roughly a second, and other services share those cores.
func (s *Store) EmbedPending(ctx context.Context, limit int) (int, error) {
	if s.emb == nil {
		return 0, nil
	}
	where, args := s.pendingFilter()
	// Cheapest pages first. The queue is ordered by page size rather than by id
	// so the hundreds of one-line database rows — facts, preferences, people,
	// documents — are all embedded before the first enormous log page starts.
	// The index becomes useful in the first minutes of a backfill that takes
	// hours to finish, and the order costs nothing: the total is the same.
	args = append(args, limit)
	rows, err := s.db.SQL().QueryContext(ctx, `
		SELECT c.id, c.page_id, p.data_source_id, p.last_edited_time, c.heading, c.text
		FROM chunks c
		JOIN pages p ON p.id = c.page_id AND p.in_trash = 0
		LEFT JOIN vectors v ON v.chunk_id = c.id
		WHERE (v.chunk_id IS NULL OR v.model <> ?)`+where+`
		ORDER BY p.content_runes ASC, c.page_id, c.ord LIMIT ?`, args...)
	if err != nil {
		return 0, err
	}
	type job struct {
		v    vector
		text string
	}
	var jobs []job
	for rows.Next() {
		var j job
		var heading, text string
		if err := rows.Scan(&j.v.chunkID, &j.v.pageID, &j.v.sourceID, &j.v.lastEdited, &heading, &text); err != nil {
			rows.Close()
			return 0, err
		}
		if heading != "" {
			text = heading + "\n" + text
		}
		j.text = text
		jobs = append(jobs, j)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return 0, err
	}

	done := 0
	for i := range jobs {
		if err := ctx.Err(); err != nil {
			return done, err
		}
		v := s.emb.EmbedPassage(jobs[i].text)
		q, scale := embed.QuantizeVector(v)
		blob := make([]byte, len(q))
		for k, x := range q {
			blob[k] = byte(x)
		}
		if _, err := s.db.SQL().ExecContext(ctx, `
			INSERT INTO vectors(chunk_id, dim, scale, vec, model) VALUES(?,?,?,?,?)
			ON CONFLICT(chunk_id) DO UPDATE SET dim=excluded.dim, scale=excluded.scale,
			  vec=excluded.vec, model=excluded.model`,
			jobs[i].v.chunkID, len(q), scale, blob, s.emb.Name()); err != nil {
			return done, err
		}
		entry := jobs[i].v
		entry.scale, entry.data = scale, q
		s.mu.Lock()
		if at, ok := s.pos[entry.chunkID]; ok {
			s.vecs[at] = entry
		} else {
			s.pos[entry.chunkID] = len(s.vecs)
			s.vecs = append(s.vecs, entry)
		}
		s.mu.Unlock()
		done++
	}
	return done, nil
}

// pendingFilter builds the shared WHERE tail for the embedding queue: the model
// name placeholder comes first, then the skip list.
func (s *Store) pendingFilter() (string, []any) {
	args := []any{s.emb.Name()}
	skip := s.skippedSources()
	if len(skip) == 0 {
		return "", args
	}
	placeholders := strings.TrimSuffix(strings.Repeat("?,", len(skip)), ",")
	for _, id := range skip {
		args = append(args, id)
	}
	return " AND p.data_source_id NOT IN (" + placeholders + ")", args
}

// StalePages lists pages whose content changed since they were last indexed.
func (s *Store) StalePages(ctx context.Context, limit int) ([]string, error) {
	rows, err := s.db.SQL().QueryContext(ctx, `
		SELECT id FROM pages
		WHERE in_trash = 0 AND indexed_hash <> content_hash
		ORDER BY last_edited_time DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out = append(out, id)
	}
	return out, rows.Err()
}

// RefreshMetadata re-reads the per-page fields the filters use. Page metadata
// changes without the chunk text changing (a moved row, a new edit time), and
// the cache would otherwise keep filtering on stale values.
func (s *Store) RefreshMetadata(ctx context.Context) error {
	rows, err := s.db.SQL().QueryContext(ctx,
		`SELECT id, data_source_id, last_edited_time, in_trash FROM pages`)
	if err != nil {
		return err
	}
	defer rows.Close()
	type meta struct {
		source, edited string
		trashed        bool
	}
	byPage := map[string]meta{}
	for rows.Next() {
		var id string
		var m meta
		var trashed int
		if err := rows.Scan(&id, &m.source, &m.edited, &trashed); err != nil {
			return err
		}
		m.trashed = trashed != 0
		byPage[id] = m
	}
	if err := rows.Err(); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	kept := s.vecs[:0]
	pos := make(map[int64]int, len(s.vecs))
	for _, v := range s.vecs {
		m, ok := byPage[v.pageID]
		if !ok || m.trashed {
			continue
		}
		v.sourceID, v.lastEdited = m.source, m.edited
		pos[v.chunkID] = len(kept)
		kept = append(kept, v)
	}
	s.vecs, s.pos = kept, pos
	return nil
}

// Stats reports index size for the status tool.
func (s *Store) Stats(ctx context.Context) (map[string]int, error) {
	out, err := s.db.Counts(ctx)
	if err != nil {
		return nil, err
	}
	pending, err := s.PendingCount(ctx)
	if err != nil {
		return nil, err
	}
	out["pending_vectors"] = pending
	out["cached_vectors"] = s.VectorCount()
	return out, nil
}

// assertDim guards against mixing vectors of different widths after a model
// change that kept the same name.
func (s *Store) assertDim(q []float32) error {
	if s.emb == nil {
		return fmt.Errorf("index: no embedding model loaded")
	}
	if len(q) != s.emb.Dim() {
		return fmt.Errorf("index: query vector has %d dims, model has %d", len(q), s.emb.Dim())
	}
	return nil
}
