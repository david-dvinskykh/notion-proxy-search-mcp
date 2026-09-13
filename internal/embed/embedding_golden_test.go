package embed

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"testing"
)

// embeddingCase is one reference vector produced by
// scripts/embedding_reference.py with onnxruntime on the checkpoint's own
// float32 ONNX export.
type embeddingCase struct {
	Mode   string    `json:"mode"`
	Text   string    `json:"text"`
	Tokens int       `json:"tokens"`
	Vector []float32 `json:"vector"`
}

func loadEmbeddingGolden(t *testing.T) []embeddingCase {
	t.Helper()
	raw, err := os.ReadFile("testdata/embedding_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []embeddingCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	return cases
}

func (c embeddingCase) embed(m *Model) []float32 {
	if c.Mode == "query" {
		return m.EmbedQuery(c.Text)
	}
	return m.EmbedPassage(c.Text)
}

// minCosine is the accuracy contract against float32. Everything below it means
// the quantized encoder is no longer the same model: at 0.99 the ranking of
// ordinary retrieval candidates is unchanged, and the measured margin on the
// real checkpoint is comfortably above it.
const minCosine = 0.99

// TestEmbeddingsMatchFloatReference compares the int8 Go encoder against
// onnxruntime running the float32 export of the same checkpoint. This is the
// check that cannot be faked by a self-consistent bug: the two paths share no
// code, only the weights.
func TestEmbeddingsMatchFloatReference(t *testing.T) {
	m := testModel(t)
	cases := loadEmbeddingGolden(t)
	if len(cases) == 0 {
		t.Fatal("golden file is empty")
	}

	worst := 1.0
	var worstText string
	for _, c := range cases {
		if len(c.Vector) != m.Dim() {
			t.Fatalf("case %q has %d dims, model has %d", c.Text, len(c.Vector), m.Dim())
		}
		got := c.embed(m)
		cos := dotFloat(got, c.Vector)
		if cos < worst {
			worst, worstText = cos, c.Text
		}
		if cos < minCosine {
			t.Errorf("cosine %.5f below %.2f for %q", cos, minCosine, c.Text)
		}
	}
	t.Logf("worst cosine against float32: %.5f (%q)", worst, truncateText(worstText, 60))
}

// tieTolerance is how close two reference scores have to be before their order
// counts as a coin toss. Quantization perturbs a score by about 2e-3; demanding
// that near-ties keep their order would be measuring noise, not quality.
const tieTolerance = 0.01

// TestQuantizedVectorsPreserveReferenceRanking is the property that actually
// matters for retrieval: after both the encoder and the stored vectors are
// quantized, the candidate the float32 model puts first must still come first,
// and deeper ranks must agree wherever the reference itself is decisive.
func TestQuantizedVectorsPreserveReferenceRanking(t *testing.T) {
	m := testModel(t)
	cases := loadEmbeddingGolden(t)

	var queries, passages []embeddingCase
	for _, c := range cases {
		if c.Mode == "query" {
			queries = append(queries, c)
			continue
		}
		passages = append(passages, c)
	}
	if len(queries) == 0 || len(passages) < 3 {
		t.Skip("golden file needs queries and at least three passages")
	}

	// Stored vectors go through int8 quantization, exactly as the index does.
	type stored struct {
		text  string
		quant []int8
		scale float32
	}
	indexed := make([]stored, 0, len(passages))
	for _, p := range passages {
		q, s := QuantizeVector(p.embed(m))
		indexed = append(indexed, stored{text: p.Text, quant: q, scale: s})
	}

	ties := 0
	for _, query := range queries {
		reference := rankByFloat(query.Vector, passages)

		qq, qs := QuantizeVector(query.embed(m))
		type scored struct {
			text  string
			score float32
		}
		got := make([]scored, 0, len(indexed))
		for _, p := range indexed {
			got = append(got, scored{p.text, DotQuantized(qq, qs, p.quant, p.scale)})
		}
		sort.SliceStable(got, func(i, j int) bool { return got[i].score > got[j].score })

		if got[0].text != reference[0].text {
			t.Errorf("query %q: top hit is %q, float32 reference says %q",
				truncateText(query.Text, 40), truncateText(got[0].text, 40),
				truncateText(reference[0].text, 40))
		}
		// Ranks two and three are still shown to a reader, so they are checked
		// too — but a swap only counts as an error when the float32 model itself
		// separated the two passages by more than the quantization noise floor.
		// The measure is the reference score of the passage that should be at
		// rank i minus the reference score of the one that landed there: that is
		// the quality actually lost, which a window around neighbouring ranks
		// overstates whenever three or four passages sit within a hundredth of
		// each other.
		refScore := make(map[string]float64, len(reference))
		for _, r := range reference {
			refScore[r.text] = r.score
		}
		for i := 1; i < 3 && i < len(reference); i++ {
			if got[i].text == reference[i].text {
				continue
			}
			lost := reference[i].score - refScore[got[i].text]
			if lost <= tieTolerance {
				ties++
				continue
			}
			t.Errorf("query %q: rank %d is %q, float32 reference says %q (reference loses %.4f)",
				truncateText(query.Text, 40), i+1, truncateText(got[i].text, 40),
				truncateText(reference[i].text, 40), lost)
		}
	}
	t.Logf("%d near-tie reorderings below the %.2f tolerance", ties, tieTolerance)
}

// TestSemanticNeighboursAreCloserThanUnrelatedText is a sanity check that does
// not depend on the reference at all: a paraphrase must beat an unrelated
// sentence. It is what catches a blob that loads and normalises correctly but
// carries scrambled weights.
func TestSemanticNeighboursAreCloserThanUnrelatedText(t *testing.T) {
	m := testModel(t)
	query := m.EmbedQuery("сколько оперативной памяти занимает контейнер")
	near := m.EmbedPassage("Контейнер gateway на Raspberry Pi держит 3,1 ГБ из 8 ГБ ОЗУ")
	far := m.EmbedPassage("Полив теплицы идёт по датчику влажности")

	nearScore, farScore := dotFloat(query, near), dotFloat(query, far)
	if nearScore <= farScore {
		t.Fatalf("paraphrase scored %.4f, unrelated text %.4f", nearScore, farScore)
	}
	t.Logf("paraphrase %.4f vs unrelated %.4f (margin %.4f)", nearScore, farScore, nearScore-farScore)

	// Cross-language retrieval is the reason a multilingual model was chosen.
	ukrainian := m.EmbedPassage("Полив теплиці працює за датчиком вологості")
	english := m.EmbedQuery("what triggers the greenhouse watering")
	if dotFloat(english, ukrainian) <= dotFloat(english, near) {
		t.Errorf("an English query should match the Ukrainian sentence about the same thing: %.4f vs %.4f",
			dotFloat(english, ukrainian), dotFloat(english, near))
	}
}

// referenceRank is one passage's float32 score for a query.
type referenceRank struct {
	text  string
	score float64
}

func rankByFloat(query []float32, passages []embeddingCase) []referenceRank {
	out := make([]referenceRank, 0, len(passages))
	for _, p := range passages {
		out = append(out, referenceRank{p.Text, dotFloat(query, p.Vector)})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].score > out[j].score })
	return out
}

func dotFloat(a, b []float32) float64 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

func truncateText(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

// BenchmarkRealModelForward measures one embedding on the installed blob. This
// is the number that sizes the indexing backlog on the target board; re-run it
// there with -benchtime 20x.
func BenchmarkRealModelForward(b *testing.B) {
	path := os.Getenv("NPS_TEST_MODEL")
	if path == "" {
		b.Skip("set NPS_TEST_MODEL to a weight blob")
	}
	m, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()

	for _, tokens := range []int{16, 40, 128, 256} {
		text := fmt.Sprintf("%s", repeatWords(tokens))
		b.Run(fmt.Sprintf("tokens%d", tokens), func(b *testing.B) {
			got := len(m.Tokenize(text))
			b.ReportMetric(float64(got), "tokens")
			for i := 0; i < b.N; i++ {
				_ = m.EmbedPassage(text)
			}
		})
	}
}

// repeatWords builds text of roughly n tokens out of single-token words.
func repeatWords(n int) string {
	out := make([]byte, 0, n*5)
	for i := 0; i < n; i++ {
		out = append(out, "слово "...)
	}
	return string(out)
}
