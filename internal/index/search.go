package index

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"unicode"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/embed"
)

// Mode selects which retrieval branches run.
type Mode string

const (
	ModeHybrid  Mode = "hybrid"
	ModeVector  Mode = "vector"
	ModeKeyword Mode = "keyword"
)

// Query is one retrieval request.
type Query struct {
	Text        string   `json:"text"`
	Limit       int      `json:"limit,omitempty"`
	Mode        Mode     `json:"mode,omitempty"`
	DataSources []string `json:"data_sources,omitempty"`
	PageIDs     []string `json:"page_ids,omitempty"`
	// EditedAfter and EditedBefore bound pages by last_edited_time (ISO 8601).
	EditedAfter  string `json:"edited_after,omitempty"`
	EditedBefore string `json:"edited_before,omitempty"`
	// ExpandRelations adds one hop of related pages behind the direct hits,
	// which is how a row's entity page comes along with the row itself.
	ExpandRelations bool `json:"expand_relations,omitempty"`
	// Candidates is how deep each branch goes before fusion.
	Candidates int `json:"candidates,omitempty"`
}

// Result is one retrieved page with its best chunk.
type Result struct {
	PageID       string  `json:"page_id"`
	Title        string  `json:"title"`
	URL          string  `json:"url"`
	DataSourceID string  `json:"data_source_id,omitempty"`
	Heading      string  `json:"heading,omitempty"`
	Snippet      string  `json:"snippet"`
	LastEdited   string  `json:"last_edited_time,omitempty"`
	Score        float64 `json:"score"`
	Keyword      float64 `json:"keyword_score,omitempty"`
	Vector       float64 `json:"vector_score,omitempty"`
	KeywordRank  int     `json:"keyword_rank,omitempty"`
	VectorRank   int     `json:"vector_rank,omitempty"`
	Via          string  `json:"via"`
	RelatedTo    string  `json:"related_to,omitempty"`
}

// rrfK is the reciprocal rank fusion constant. 60 is the value the original
// TREC work settled on and what every hybrid retriever since has used; it makes
// the top of each list matter without letting one branch dominate.
const rrfK = 60.0

// Branch weights for fusion. Plain unweighted RRF is wrong here, and the
// measurement says so: on a 45-document corpus of Russian and Ukrainian notes,
// vector search answered 81 % of questions at rank one and 100 % within three,
// BM25 managed 51 % and 70 % — and equal-weight fusion came out at 78 % / 95 %,
// below the vector branch it was supposed to improve. A
// rank-one keyword hit on the wrong page scores 1/61 and outranks a correct
// vector hit at rank three (1/63), so the weaker branch was dragging the
// stronger one down.
//
// Keyword is kept rather than dropped because it answers what embeddings
// cannot: exact identifiers, numbers, file names, environment variables. That
// is not a hunch either — see the identifier column below, where taking the
// weight to zero costs a question and 0.11 of MRR.
//
// The weight is 0.20 because the live mirror says so. The first sweep ran on
// the 45-document corpus and chose 0.35 off a flat 0.15-0.35 plateau, taking
// the upper end on the argument that a larger corpus would need more help
// promoting literals. The larger corpus arrived — 6 500 chunks of the real
// workspace — and says the opposite: the plateau moved down, and 0.35 now sits
// just past its edge. 27 questions, 20 natural-language and 7 literal
// identifiers whose answer page is unique in the mirror:
//
//	w_kw   all hit@1  all MRR   natural MRR   identifier hit@1  identifier MRR
//	0.00     17/27     0.780       0.793           4/7              0.743
//	0.15     18/27     0.805       0.787           5/7              0.857
//	0.20     18/27     0.803       0.784           5/7              0.857
//	0.35     17/27     0.783       0.757           5/7              0.857
//	1.00     17/27     0.757       0.698           6/7              0.929
//
// Identifier lookups are flat from 0.10 to 0.50, so the choice inside the
// plateau is decided by the natural-language half, which keeps losing ground as
// the weight rises. 0.20 is the top of the plateau: the most help the keyword
// branch can give a literal before it starts demoting correct vector hits.
// Re-run the sweep before changing this; it is offline arithmetic over the two
// branch rankings and needs no rebuild.
const (
	weightVector  = 1.0
	weightKeyword = 0.20
)

// Candidate pool sizing. Both branches hand fusion a list of pages, so the
// keyword branch has to read more chunks than the pages it is asked for.
const (
	chunkOversample = 25
	maxKeywordScan  = 5000
)

// Search runs the requested branches and fuses them.
func (s *Store) Search(ctx context.Context, q Query) ([]Result, error) {
	if q.Limit <= 0 {
		q.Limit = 10
	}
	if q.Candidates <= 0 {
		q.Candidates = q.Limit * 5
		if q.Candidates < 40 {
			q.Candidates = 40
		}
	}
	mode := q.Mode
	if mode == "" {
		mode = ModeHybrid
	}
	if (mode == ModeHybrid || mode == ModeVector) && s.emb == nil {
		// Degrading to keyword search is better than failing: the stdio
		// process can run without a model installed.
		mode = ModeKeyword
	}

	var keyword, vector []hit
	var err error
	if mode == ModeKeyword || mode == ModeHybrid {
		keyword, err = s.keywordHits(ctx, q)
		if err != nil {
			return nil, err
		}
	}
	if mode == ModeVector || mode == ModeHybrid {
		vector, err = s.vectorHits(ctx, q)
		if err != nil {
			return nil, err
		}
	}

	fused := fuse(keyword, vector)
	sort.SliceStable(fused, func(i, j int) bool { return fused[i].score > fused[j].score })
	if len(fused) > q.Limit {
		fused = fused[:q.Limit]
	}

	results, err := s.materialize(ctx, fused)
	if err != nil {
		return nil, err
	}
	if q.ExpandRelations {
		extra, err := s.expand(ctx, results, q)
		if err != nil {
			return nil, err
		}
		results = append(results, extra...)
	}
	return results, nil
}

// hit is one branch's scored chunk.
type hit struct {
	chunkID int64
	pageID  string
	score   float64
	rank    int
}

// fused is a page after reciprocal rank fusion.
type fusedHit struct {
	pageID               string
	chunkID              int64
	score                float64
	keyword, vector      float64
	keywordRank, vecRank int
}

// fuse combines branch rankings with reciprocal rank fusion, keeping the best
// chunk per page so one long page cannot fill the whole answer.
//
// Each branch contributes exactly once per page, from that page's best rank in
// it. Summing every chunk instead would rank by chunk count: a 770-chunk
// reference page collects dozens of mediocre hits and beats the one-sentence
// fact that actually answers the question and stands first in the branch. That
// is not hypothetical — it is what the mirror did on this workspace, where a
// fact with vector rank 1 came back below a page whose best chunk was rank 15.
func fuse(keyword, vector []hit) []fusedHit {
	byPage := map[string]*fusedHit{}
	order := []string{}
	apply := func(list []hit, isVector bool) {
		for _, h := range list {
			f, ok := byPage[h.pageID]
			if !ok {
				f = &fusedHit{pageID: h.pageID, chunkID: h.chunkID}
				byPage[h.pageID] = f
				order = append(order, h.pageID)
			}
			if isVector {
				if f.vecRank == 0 || h.rank < f.vecRank {
					f.vecRank, f.vector = h.rank, h.score
					if f.keywordRank == 0 || h.rank <= f.keywordRank {
						f.chunkID = h.chunkID
					}
				}
				continue
			}
			if f.keywordRank == 0 || h.rank < f.keywordRank {
				f.keywordRank, f.keyword = h.rank, h.score
				f.chunkID = h.chunkID
			}
		}
	}
	apply(keyword, false)
	apply(vector, true)

	out := make([]fusedHit, 0, len(order))
	for _, id := range order {
		f := *byPage[id]
		f.score = 0
		if f.vecRank > 0 {
			f.score += weightVector / (rrfK + float64(f.vecRank))
		}
		if f.keywordRank > 0 {
			f.score += weightKeyword / (rrfK + float64(f.keywordRank))
		}
		out = append(out, f)
	}
	return out
}

// keywordHits runs FTS5 BM25 over the chunk text.
func (s *Store) keywordHits(ctx context.Context, q Query) ([]hit, error) {
	match := BuildMatchExpression(q.Text)
	if match == "" {
		return nil, nil
	}
	where := []string{"chunks_fts MATCH ?", "p.in_trash = 0"}
	args := []any{match}
	where, args = appendFilters(where, args, q)

	// The BM25 column weights put a hit in the title above one in the body:
	// a page named after the thing asked about is nearly always the answer.
	query := `
		SELECT c.id, c.page_id, -bm25(chunks_fts, 1.0, 1.5, 3.0) AS score
		FROM chunks_fts
		JOIN chunks c ON c.id = chunks_fts.rowid
		JOIN pages p ON p.id = c.page_id
		WHERE ` + strings.Join(where, " AND ") + `
		ORDER BY score DESC LIMIT ?`
	args = append(args, keywordScanDepth(q.Candidates))

	rows, err := s.db.SQL().QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("index: keyword search: %w", err)
	}
	defer rows.Close()
	var out []hit
	for rows.Next() {
		var h hit
		if err := rows.Scan(&h.chunkID, &h.pageID, &h.score); err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return topPages(out, q.Candidates), nil
}

// keywordScanDepth is how many chunks the BM25 query reads before the result is
// collapsed to one row per page. It has to overshoot the page budget, because a
// long page answers MATCH with one row per chunk and would otherwise fill the
// whole window on its own; the cap keeps a query that matches half the mirror
// from reading half the mirror.
func keywordScanDepth(candidates int) int {
	depth := candidates * chunkOversample
	if depth > maxKeywordScan {
		depth = maxKeywordScan
	}
	return depth
}

// topPages collapses a score-ordered chunk list to its best chunk per page and
// ranks what is left by page, not by chunk.
//
// Ranking chunks is what made a single long page able to push everything else
// out of a branch before fusion ever saw it: 40 candidate slots against a page
// that owns 770 chunks is not a contest. Ranks handed to fusion are page ranks
// from here on, so rank 3 means the third best page rather than the third best
// paragraph of the first one.
func topPages(hits []hit, limit int) []hit {
	if limit <= 0 {
		return nil
	}
	seen := make(map[string]bool, limit)
	out := make([]hit, 0, limit)
	for _, h := range hits {
		if seen[h.pageID] {
			continue
		}
		seen[h.pageID] = true
		h.rank = len(out) + 1
		out = append(out, h)
		if len(out) == limit {
			break
		}
	}
	return out
}

func appendFilters(where []string, args []any, q Query) ([]string, []any) {
	if len(q.DataSources) > 0 {
		where = append(where, "p.data_source_id IN ("+placeholders(len(q.DataSources))+")")
		for _, ds := range q.DataSources {
			args = append(args, ds)
		}
	}
	if len(q.PageIDs) > 0 {
		where = append(where, "p.id IN ("+placeholders(len(q.PageIDs))+")")
		for _, id := range q.PageIDs {
			args = append(args, id)
		}
	}
	if q.EditedAfter != "" {
		where = append(where, "p.last_edited_time >= ?")
		args = append(args, q.EditedAfter)
	}
	if q.EditedBefore != "" {
		where = append(where, "p.last_edited_time <= ?")
		args = append(args, q.EditedBefore)
	}
	return where, args
}

func placeholders(n int) string {
	return strings.TrimSuffix(strings.Repeat("?,", n), ",")
}

// vectorHits scans the in-memory int8 vectors. A brute-force scan over tens of
// thousands of 384-byte vectors costs a few milliseconds and needs no index to
// rebuild; an approximate structure would only pay off an order of magnitude
// higher.
func (s *Store) vectorHits(ctx context.Context, q Query) ([]hit, error) {
	qv := s.emb.EmbedQuery(q.Text)
	if err := s.assertDim(qv); err != nil {
		return nil, err
	}
	qq, qs := embed.QuantizeVector(qv)

	allowSource := setOf(q.DataSources)
	allowPage := setOf(q.PageIDs)

	s.mu.RLock()
	scored := make([]hit, 0, len(s.vecs))
	for i := range s.vecs {
		v := &s.vecs[i]
		if allowSource != nil && !allowSource[v.sourceID] {
			continue
		}
		if allowPage != nil && !allowPage[v.pageID] {
			continue
		}
		if q.EditedAfter != "" && v.lastEdited < q.EditedAfter {
			continue
		}
		if q.EditedBefore != "" && v.lastEdited > q.EditedBefore {
			continue
		}
		if len(v.data) != len(qq) {
			continue
		}
		scored = append(scored, hit{
			chunkID: v.chunkID,
			pageID:  v.pageID,
			score:   float64(embed.DotQuantized(qq, qs, v.data, v.scale)),
		})
	}
	s.mu.RUnlock()

	sort.Slice(scored, func(i, j int) bool { return scored[i].score > scored[j].score })
	return topPages(scored, q.Candidates), nil
}

func setOf(values []string) map[string]bool {
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]bool, len(values))
	for _, v := range values {
		out[v] = true
	}
	return out
}

// materialize turns fused page hits into results with text and metadata.
func (s *Store) materialize(ctx context.Context, fused []fusedHit) ([]Result, error) {
	out := make([]Result, 0, len(fused))
	for _, f := range fused {
		row := s.db.SQL().QueryRowContext(ctx, `
			SELECT p.title, p.url, p.data_source_id, p.last_edited_time, c.heading, c.text
			FROM chunks c JOIN pages p ON p.id = c.page_id WHERE c.id = ?`, f.chunkID)
		var r Result
		if err := row.Scan(&r.Title, &r.URL, &r.DataSourceID, &r.LastEdited, &r.Heading, &r.Snippet); err != nil {
			// A chunk deleted between the scan and here is not an error.
			continue
		}
		r.PageID = f.pageID
		r.Score = f.score
		r.Keyword, r.Vector = f.keyword, f.vector
		r.KeywordRank, r.VectorRank = f.keywordRank, f.vecRank
		switch {
		case f.keywordRank > 0 && f.vecRank > 0:
			r.Via = "both"
		case f.vecRank > 0:
			r.Via = "vector"
		default:
			r.Via = "keyword"
		}
		out = append(out, r)
	}
	return out, nil
}

// expand adds pages one relation hop away from the direct hits. The constitution
// reads a fact and then wants the entity card it hangs off; this is that hop,
// done in SQL instead of a second round trip.
func (s *Store) expand(ctx context.Context, seeds []Result, q Query) ([]Result, error) {
	if len(seeds) == 0 {
		return nil, nil
	}
	seen := make(map[string]bool, len(seeds))
	ids := make([]any, 0, len(seeds))
	for _, r := range seeds {
		seen[r.PageID] = true
		ids = append(ids, r.PageID)
	}
	limit := q.Limit
	if limit > 10 {
		limit = 10
	}

	query := `
		SELECT r.from_id, r.to_id, p.id, p.title, p.url, p.data_source_id, p.last_edited_time
		FROM relations r
		JOIN pages p ON p.id = CASE WHEN r.from_id IN (` + placeholders(len(ids)) + `) THEN r.to_id ELSE r.from_id END
		WHERE (r.from_id IN (` + placeholders(len(ids)) + `) OR r.to_id IN (` + placeholders(len(ids)) + `))
		  AND p.in_trash = 0
		LIMIT ?`
	args := append(append(append([]any{}, ids...), ids...), ids...)
	args = append(args, limit*4)

	rows, err := s.db.SQL().QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("index: relation expansion: %w", err)
	}
	defer rows.Close()

	byScore := map[string]float64{}
	for _, r := range seeds {
		byScore[r.PageID] = r.Score
	}
	var out []Result
	for rows.Next() && len(out) < limit {
		var fromID, toID string
		var r Result
		if err := rows.Scan(&fromID, &toID, &r.PageID, &r.Title, &r.URL, &r.DataSourceID, &r.LastEdited); err != nil {
			return nil, err
		}
		if seen[r.PageID] {
			continue
		}
		seen[r.PageID] = true
		parent := fromID
		if byScore[toID] > byScore[fromID] {
			parent = toID
		}
		r.Via = "relation"
		r.RelatedTo = parent
		// A related page is context, not an answer: a third of its parent's
		// score keeps it below every direct hit.
		r.Score = byScore[parent] / 3
		r.Snippet = r.Title
		out = append(out, r)
	}
	return out, rows.Err()
}

// BuildMatchExpression turns free text into an FTS5 MATCH expression. Terms are
// ORed so a long question still matches, and terms of four characters or more
// get a prefix wildcard, which is the cheapest stand-in for Russian and
// Ukrainian morphology that SQLite's unicode61 tokenizer lacks.
func BuildMatchExpression(text string) string {
	var terms []string
	for _, word := range strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
	}) {
		word = strings.ToLower(word)
		if len([]rune(word)) < 2 {
			continue
		}
		quoted := `"` + strings.ReplaceAll(word, `"`, `""`) + `"`
		if len([]rune(word)) >= 4 {
			quoted += "*"
		}
		terms = append(terms, quoted)
		if len(terms) == 32 {
			break
		}
	}
	return strings.Join(terms, " OR ")
}
