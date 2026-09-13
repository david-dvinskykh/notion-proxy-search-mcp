package index

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/embed"
)

// evalFact is one indexed document in an evaluation corpus.
type evalFact struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

// evalQuestion is one query with the documents that answer it, best first.
type evalQuestion struct {
	Q    string   `json:"q"`
	Want []string `json:"want"`
}

// TestRetrievalQualityOnCorpus measures hit rates for the three retrieval modes
// on a real corpus. It is the only way to tell whether a ranking change helps,
// so it is committed even though the corpus is not: point it at your own data.
//
//	NPS_TEST_MODEL=model.npse \
//	NPS_EVAL_FACTS=facts.json NPS_EVAL_QUESTIONS=questions.json \
//	go test ./internal/index/ -run RetrievalQuality -v
//
// facts.json is [{"id","text"}...]; questions.json is [{"q","want":[id...]}...]
// where want lists every acceptable answer. Corpora drawn from a private
// workspace stay out of the repository.
func TestRetrievalQualityOnCorpus(t *testing.T) {
	factsPath := os.Getenv("NPS_EVAL_FACTS")
	questionsPath := os.Getenv("NPS_EVAL_QUESTIONS")
	modelPath := os.Getenv("NPS_TEST_MODEL")
	if factsPath == "" || questionsPath == "" {
		t.Skip("set NPS_EVAL_FACTS and NPS_EVAL_QUESTIONS to run the retrieval evaluation")
	}

	var facts []evalFact
	readJSON(t, factsPath, &facts)
	var questions []evalQuestion
	readJSON(t, questionsPath, &questions)
	if len(facts) == 0 || len(questions) == 0 {
		t.Fatal("evaluation corpus is empty")
	}

	var embedder Embedder
	if modelPath != "" {
		model, err := embed.Load(modelPath)
		if err != nil {
			t.Fatalf("load model: %v", err)
		}
		defer model.Close()
		embedder = model
	}

	ctx := context.Background()
	f := newFixture(t, embedder)
	f.addSource(t, "facts", `{"Утверждение":{"type":"title"}}`)

	indexStart := time.Now()
	for _, fact := range facts {
		f.addPage(t, fact.ID, "facts", fact.Text, "", "", "2026-09-01T00:00:00.000Z")
	}
	chunked := time.Since(indexStart)

	embedStart := time.Now()
	embedded := 0
	if embedder != nil {
		n, err := f.store.EmbedPending(ctx, len(facts)*4)
		if err != nil {
			t.Fatal(err)
		}
		embedded = n
	}
	embedTime := time.Since(embedStart)

	t.Logf("corpus: %d facts, mirrored and chunked in %s, %d chunks embedded in %s",
		len(facts), chunked.Round(time.Millisecond), embedded, embedTime.Round(time.Millisecond))
	if embedded > 0 {
		t.Logf("embedding: %s per chunk on %d-core amd64", (embedTime / time.Duration(embedded)).Round(time.Millisecond), 4)
	}

	modes := []Mode{ModeKeyword}
	if embedder != nil {
		modes = append(modes, ModeVector, ModeHybrid)
	}

	type tally struct {
		hit1, hit3, latency int
		elapsed             time.Duration
		misses              []string
	}
	results := map[Mode]*tally{}

	for _, mode := range modes {
		score := &tally{}
		for _, question := range questions {
			start := time.Now()
			got, err := f.store.Search(ctx, Query{Text: question.Q, Limit: 3, Mode: mode})
			if err != nil {
				t.Fatalf("search %q: %v", question.Q, err)
			}
			score.elapsed += time.Since(start)
			score.latency++

			want := make(map[string]bool, len(question.Want))
			for _, id := range question.Want {
				want[id] = true
			}
			if len(got) > 0 && want[got[0].PageID] {
				score.hit1++
			} else {
				shown := "nothing"
				if len(got) > 0 {
					shown = got[0].PageID
				}
				score.misses = append(score.misses,
					fmt.Sprintf("%q -> %s (want %v)", question.Q, shown, question.Want))
			}
			for i, r := range got {
				if i < 3 && want[r.PageID] {
					score.hit3++
					break
				}
			}
		}
		results[mode] = score
	}

	sort.Slice(modes, func(i, j int) bool { return modes[i] < modes[j] })
	for _, mode := range modes {
		s := results[mode]
		t.Logf("%-8s hit@1 %2d/%d (%.0f%%)  hit@3 %2d/%d (%.0f%%)  median search %s",
			mode, s.hit1, len(questions), 100*float64(s.hit1)/float64(len(questions)),
			s.hit3, len(questions), 100*float64(s.hit3)/float64(len(questions)),
			(s.elapsed / time.Duration(s.latency)).Round(time.Microsecond))
	}
	for _, mode := range modes {
		for _, miss := range results[mode].misses {
			t.Logf("  %s miss: %s", mode, miss)
		}
	}

	// The gate: hybrid must be at least as good as either branch alone. A
	// ranking change that breaks this is a regression however good it looks.
	if embedder != nil {
		hybrid := results[ModeHybrid].hit3
		if hybrid < results[ModeKeyword].hit3 || hybrid < results[ModeVector].hit3 {
			t.Errorf("hybrid hit@3 %d is below a single branch (keyword %d, vector %d)",
				hybrid, results[ModeKeyword].hit3, results[ModeVector].hit3)
		}
	}
}

func readJSON(t *testing.T, path string, into any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, into); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}
