package index

import (
	"context"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mirror"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/notion"
)

// conceptEmbedder is a deterministic stand-in for the real model: it maps words
// to concept dimensions, so a paraphrase that shares no literal token with the
// query still lands nearby. That is exactly the behaviour vector search is
// bought for, and it can be asserted without loading weights.
type conceptEmbedder struct{}

var concepts = map[string]int{
	// memory / RAM
	"память": 0, "памяти": 0, "озу": 0, "ram": 0, "гб": 0, "потребление": 0, "ест": 0,
	// container / gateway
	"контейнер": 1, "gateway": 1, "стек": 1, "docker": 1,
	// greenhouse watering
	"полив": 2, "влажность": 2, "влажности": 2, "теплица": 2, "теплицы": 2, "датчик": 2, "датчику": 2,
	// money
	"деньги": 3, "платёж": 3, "расход": 3, "счёт": 3,
}

func (conceptEmbedder) Dim() int     { return 8 }
func (conceptEmbedder) Name() string { return "concept-test" }

func (c conceptEmbedder) vector(text string) []float32 {
	out := make([]float32, 8)
	for _, w := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !('а' <= r && r <= 'я') && !('a' <= r && r <= 'z') && !('0' <= r && r <= '9') && r != 'ё'
	}) {
		if dim, ok := concepts[w]; ok {
			out[dim] += 1
		}
	}
	out[7] += 0.1 // keeps an all-unknown text from being the zero vector
	var norm float64
	for _, v := range out {
		norm += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(norm))
	for i := range out {
		out[i] *= inv
	}
	return out
}

func (c conceptEmbedder) EmbedQuery(text string) []float32   { return c.vector(text) }
func (c conceptEmbedder) EmbedPassage(text string) []float32 { return c.vector(text) }

type fixture struct {
	db    *mirror.DB
	store *Store
}

func newFixture(t *testing.T, emb Embedder) *fixture {
	t.Helper()
	db, err := mirror.Open(context.Background(), filepath.Join(t.TempDir(), "mirror.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return &fixture{db: db, store: New(db, emb)}
}

func (f *fixture) addSource(t *testing.T, id, schema string) {
	t.Helper()
	ds := &notion.DataSource{Schema: []byte(schema)}
	ds.ID = id
	if err := f.db.UpsertDataSource(context.Background(), ds); err != nil {
		t.Fatal(err)
	}
}

func (f *fixture) addPage(t *testing.T, id, sourceID, title, markdown, props, edited string) {
	t.Helper()
	ctx := context.Background()
	p := &notion.Page{}
	p.ID = id
	p.Object.Object = "page"
	p.URL = "https://app.notion.com/p/" + id
	p.LastEditedTime = edited
	if props == "" {
		props = `{"Name":{"type":"title","title":[{"type":"text","plain_text":` + quote(title) + `,"annotations":{}}]}}`
	}
	p.Properties = []byte(props)
	if _, err := f.db.UpsertPage(ctx, p, sourceID); err != nil {
		t.Fatal(err)
	}
	if err := f.db.SetContent(ctx, id, markdown); err != nil {
		t.Fatal(err)
	}
	if err := f.store.Reindex(ctx, id); err != nil {
		t.Fatal(err)
	}
}

func quote(s string) string { return `"` + strings.ReplaceAll(s, `"`, `\"`) + `"` }

func (f *fixture) embedAll(t *testing.T) {
	t.Helper()
	if _, err := f.store.EmbedPending(context.Background(), 1000); err != nil {
		t.Fatal(err)
	}
}

func seedFacts(t *testing.T, f *fixture) {
	f.addSource(t, "facts", `{"Утверждение":{"type":"title"},"Доверие":{"type":"select"}}`)
	f.addPage(t, "fact-ram", "facts",
		"Контейнер gateway на Raspberry Pi держит 3,1 ГБ ОЗУ",
		"Замер 04.09.2026: usage 3 271 659 520 байт при лимите 8 ГБ.\n", "", "2026-09-04T14:47:00.000Z")
	f.addPage(t, "fact-water", "facts",
		"Полив теплицы идёт по датчику влажности",
		"Полив включается по влажности почвы.\n", "", "2026-08-20T10:00:00.000Z")
	f.addPage(t, "fact-money", "facts",
		"Расходы по проекту разносятся по категориям",
		"Перед правкой открыть процесс целиком.\n", "", "2026-09-01T10:00:00.000Z")
}

func titlesOf(results []Result) []string {
	out := make([]string, 0, len(results))
	for _, r := range results {
		out = append(out, r.Title)
	}
	return out
}

func TestKeywordSearchFindsLiteralMatch(t *testing.T) {
	f := newFixture(t, nil)
	seedFacts(t, f)

	got, err := f.store.Search(context.Background(), Query{Text: "gateway ОЗУ", Limit: 3, Mode: ModeKeyword})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || got[0].PageID != "fact-ram" {
		t.Fatalf("expected the RAM fact first, got %v", titlesOf(got))
	}
	if got[0].Via != "keyword" {
		t.Fatalf("via = %q", got[0].Via)
	}
}

// TestVectorSearchFindsParaphrase is the reason the embedding engine exists: the
// query shares no searchable token with the page that answers it.
func TestVectorSearchFindsParaphrase(t *testing.T) {
	f := newFixture(t, conceptEmbedder{})
	seedFacts(t, f)
	f.embedAll(t)

	keywordOnly, err := f.store.Search(context.Background(),
		Query{Text: "сколько ест докер", Limit: 3, Mode: ModeKeyword})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range keywordOnly {
		if r.PageID == "fact-ram" {
			t.Skip("the keyword branch already matches; the paraphrase is not blind")
		}
	}

	got, err := f.store.Search(context.Background(),
		Query{Text: "сколько ест докер", Limit: 3, Mode: ModeVector})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || got[0].PageID != "fact-ram" {
		t.Fatalf("vector search missed the paraphrase, got %v", titlesOf(got))
	}
}

func TestHybridBeatsEitherBranchAlone(t *testing.T) {
	f := newFixture(t, conceptEmbedder{})
	seedFacts(t, f)
	f.embedAll(t)

	got, err := f.store.Search(context.Background(),
		Query{Text: "потребление памяти контейнером gateway", Limit: 3, Mode: ModeHybrid})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || got[0].PageID != "fact-ram" {
		t.Fatalf("got %v", titlesOf(got))
	}
	if got[0].Via != "both" {
		t.Fatalf("a page found by both branches must be marked as such, got %q", got[0].Via)
	}
	if got[0].KeywordRank == 0 || got[0].VectorRank == 0 {
		t.Fatalf("per-branch ranks missing: %+v", got[0])
	}
}

// TestLongPageDoesNotOutrankBetterShortOne pins the counting rule of the
// fusion: a page scores from its best rank in each branch, once. Summing every
// chunk instead ranks by page length, since a reference page contributes one
// term per chunk. Here the long page is worse in both branches at every rank
// and must still lose — on the live workspace it did not, and a one-sentence
// fact standing first by cosine came back below a page whose best chunk was
// fifteenth.
func TestLongPageDoesNotOutrankBetterShortOne(t *testing.T) {
	short := []hit{{pageID: "fact", chunkID: 1, rank: 1, score: 0.86}}
	long := make([]hit, 0, 40)
	for i := 0; i < 40; i++ {
		long = append(long, hit{pageID: "manual", chunkID: int64(100 + i), rank: 2 + i, score: 0.80})
	}

	out := fuse(nil, append(short, long...))
	byPage := map[string]fusedHit{}
	for _, f := range out {
		byPage[f.pageID] = f
	}
	if byPage["fact"].score <= byPage["manual"].score {
		t.Fatalf("40 mediocre chunks outscored the top hit: fact %.5f, manual %.5f",
			byPage["fact"].score, byPage["manual"].score)
	}
	if byPage["manual"].vecRank != 2 {
		t.Fatalf("long page kept rank %d instead of its best", byPage["manual"].vecRank)
	}
}

func TestSearchFiltersByDataSource(t *testing.T) {
	f := newFixture(t, conceptEmbedder{})
	seedFacts(t, f)
	f.addSource(t, "journal", `{"Дата":{"type":"title"}}`)
	f.addPage(t, "day-1", "journal", "2026-09-04",
		"Контейнер gateway проверен, ОЗУ в норме.\n", "", "2026-09-04T21:00:00.000Z")
	f.embedAll(t)

	got, err := f.store.Search(context.Background(), Query{
		Text: "gateway ОЗУ", Limit: 5, Mode: ModeHybrid, DataSources: []string{"journal"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 {
		t.Fatal("no results")
	}
	for _, r := range got {
		if r.DataSourceID != "journal" {
			t.Fatalf("filter leaked a row from %q", r.DataSourceID)
		}
	}
}

func TestSearchFiltersByEditedWindow(t *testing.T) {
	f := newFixture(t, conceptEmbedder{})
	seedFacts(t, f)
	f.embedAll(t)

	got, err := f.store.Search(context.Background(), Query{
		Text: "память деньги полив", Limit: 10, Mode: ModeHybrid, EditedAfter: "2026-09-01",
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range got {
		if r.LastEdited < "2026-09-01" {
			t.Fatalf("page %s edited %s slipped through the window", r.PageID, r.LastEdited)
		}
	}
	if len(got) == 0 {
		t.Fatal("expected the September facts")
	}
}

func TestRelationExpansionAddsContextBelowDirectHits(t *testing.T) {
	f := newFixture(t, conceptEmbedder{})
	f.addSource(t, "facts", `{"Утверждение":{"type":"title"},"День":{"type":"relation"}}`)
	f.addSource(t, "journal", `{"Дата":{"type":"title"}}`)
	f.addPage(t, "day-1", "journal", "2026-09-04", "Карточка дня.\n", "", "2026-09-04T21:00:00.000Z")
	f.addPage(t, "fact-ram", "facts", "Контейнер gateway держит 3,1 ГБ ОЗУ",
		"Замер.\n",
		`{"Утверждение":{"type":"title","title":[{"type":"text","plain_text":"Контейнер gateway держит 3,1 ГБ ОЗУ","annotations":{}}]},
		  "День":{"type":"relation","relation":[{"id":"day-1"}]}}`,
		"2026-09-04T14:47:00.000Z")
	f.embedAll(t)

	got, err := f.store.Search(context.Background(), Query{
		Text: "gateway ОЗУ", Limit: 3, Mode: ModeKeyword, ExpandRelations: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	var related *Result
	for i := range got {
		if got[i].Via == "relation" {
			related = &got[i]
		}
	}
	if related == nil {
		t.Fatalf("relation hop missing from %v", titlesOf(got))
	}
	if related.PageID != "day-1" || related.RelatedTo != "fact-ram" {
		t.Fatalf("unexpected related row %+v", related)
	}
	if related.Score >= got[0].Score {
		t.Fatalf("context must rank below the direct hit: %v vs %v", related.Score, got[0].Score)
	}
}

func TestReindexReusesUnchangedChunkVectors(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t, conceptEmbedder{})
	f.addSource(t, "docs", `{"Name":{"type":"title"}}`)
	f.addPage(t, "doc-1", "docs", "Процесс",
		"## Шаг 1\n\nПервый шаг.\n\n## Шаг 2\n\nВторой шаг.\n", "", "2026-09-01T00:00:00.000Z")
	f.embedAll(t)

	before, err := f.store.PendingCount(ctx)
	if err != nil || before != 0 {
		t.Fatalf("pending = %d, err = %v", before, err)
	}

	// Edit only the second section.
	if err := f.db.SetContent(ctx, "doc-1",
		"## Шаг 1\n\nПервый шаг.\n\n## Шаг 2\n\nВторой шаг, переписанный.\n"); err != nil {
		t.Fatal(err)
	}
	if err := f.store.Reindex(ctx, "doc-1"); err != nil {
		t.Fatal(err)
	}
	pending, err := f.store.PendingCount(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if pending != 1 {
		t.Fatalf("pending = %d, want exactly the changed chunk", pending)
	}
}

func TestTrashedPagesLeaveTheIndex(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t, conceptEmbedder{})
	seedFacts(t, f)
	f.embedAll(t)

	if err := f.db.MarkTrashed(ctx, "fact-ram"); err != nil {
		t.Fatal(err)
	}
	if err := f.store.Reindex(ctx, "fact-ram"); err != nil {
		t.Fatal(err)
	}
	got, err := f.store.Search(ctx, Query{Text: "gateway ОЗУ", Limit: 5, Mode: ModeHybrid})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range got {
		if r.PageID == "fact-ram" {
			t.Fatal("a trashed page is still searchable")
		}
	}
}

func TestBuildMatchExpression(t *testing.T) {
	got := BuildMatchExpression("Память и ОЗУ, 3.1 ГБ!")
	for _, want := range []string{`"память"*`, `"озу"`, `"гб"`, " OR "} {
		if !strings.Contains(got, want) {
			t.Fatalf("expression %q lacks %q", got, want)
		}
	}
	if BuildMatchExpression("! ?") != "" {
		t.Fatal("punctuation-only input must produce no expression")
	}
}

func TestSearchWithoutEmbedderFallsBackToKeyword(t *testing.T) {
	f := newFixture(t, nil)
	seedFacts(t, f)
	got, err := f.store.Search(context.Background(), Query{Text: "gateway", Limit: 3, Mode: ModeHybrid})
	if err != nil {
		t.Fatalf("hybrid search must degrade, not fail: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("no results from the keyword fallback")
	}
}

// TestEmbeddingQueueTakesCheapestPagesFirst guards the ordering that makes a
// long backfill useful early: hundreds of one-line database rows must all be
// embedded before the first enormous page starts.
func TestEmbeddingQueueTakesCheapestPagesFirst(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t, conceptEmbedder{})
	f.addSource(t, "journal", `{"Дата":{"type":"title"}}`)
	f.addSource(t, "facts", `{"Утверждение":{"type":"title"}}`)

	// One journal page the size of a real day card, and three short facts.
	f.addPage(t, "day-1", "journal", "2026-09-12",
		strings.Repeat("Событие дня, подробности и замеры. ", 400), "", "2026-09-12T21:00:00.000Z")
	for _, id := range []string{"fact-a", "fact-b", "fact-c"} {
		f.addPage(t, id, "facts", "Короткий факт "+id, "Одна строка.\n", "", "2026-09-01T00:00:00.000Z")
	}

	// A batch smaller than the day card's chunk count must still cover the facts.
	if _, err := f.store.EmbedPending(ctx, 3); err != nil {
		t.Fatal(err)
	}
	var embeddedFacts int
	if err := f.db.SQL().QueryRowContext(ctx, `
		SELECT COUNT(*) FROM vectors v JOIN chunks c ON c.id = v.chunk_id
		JOIN pages p ON p.id = c.page_id WHERE p.data_source_id = 'facts'`).Scan(&embeddedFacts); err != nil {
		t.Fatal(err)
	}
	if embeddedFacts != 3 {
		t.Fatalf("embedded %d of 3 short facts first; the day card jumped the queue", embeddedFacts)
	}
}

// TestSkippedSourcesStayOutOfTheVectorIndex covers the escape hatch for bulk
// pages that are not worth embedding: they keep working through keyword search.
func TestSkippedSourcesStayOutOfTheVectorIndex(t *testing.T) {
	ctx := context.Background()
	f := newFixture(t, conceptEmbedder{})
	seedFacts(t, f)
	f.addSource(t, "journal", `{"Дата":{"type":"title"}}`)
	f.addPage(t, "day-1", "journal", "2026-09-04",
		"Контейнер gateway проверен, ОЗУ в норме.\n", "", "2026-09-04T21:00:00.000Z")

	f.store.SkipSources([]string{"journal"})
	pending, err := f.store.PendingCount(ctx)
	if err != nil {
		t.Fatal(err)
	}
	f.embedAll(t)
	if after, err := f.store.PendingCount(ctx); err != nil || after != 0 {
		t.Fatalf("pending after embedding = %d (err %v), was %d", after, err, pending)
	}

	var journalVectors int
	if err := f.db.SQL().QueryRowContext(ctx, `
		SELECT COUNT(*) FROM vectors v JOIN chunks c ON c.id = v.chunk_id
		JOIN pages p ON p.id = c.page_id WHERE p.data_source_id = 'journal'`).Scan(&journalVectors); err != nil {
		t.Fatal(err)
	}
	if journalVectors != 0 {
		t.Fatalf("skipped source got %d vectors", journalVectors)
	}

	// Keyword search must still reach it.
	got, err := f.store.Search(ctx, Query{Text: "gateway ОЗУ", Limit: 5, Mode: ModeKeyword})
	if err != nil {
		t.Fatal(err)
	}
	var found bool
	for _, r := range got {
		if r.PageID == "day-1" {
			found = true
		}
	}
	if !found {
		t.Fatalf("a skipped page must stay findable by keyword: %v", titlesOf(got))
	}
}
