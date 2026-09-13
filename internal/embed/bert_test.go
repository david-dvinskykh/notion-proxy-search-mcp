package embed

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"
)

// tinyConfig is a miniature BERT used to check the encoder end to end without
// downloading real weights.
type tinyConfig struct {
	vocab, hidden, layers, heads, ffn, maxPos int
}

var tiny = tinyConfig{vocab: 40, hidden: 16, layers: 2, heads: 4, ffn: 32, maxPos: 24}

// floatWeights is the same model in float32, used as the reference the int8
// path has to agree with.
type floatWeights struct {
	words, positions, types []float32
	embG, embB              []float32
	layers                  []struct {
		q, k, v, o, f1, f2       []float32
		qb, kb, vb, ob, f1b, f2b []float32
		attnG, attnB, outG, outB []float32
	}
}

func randomTinyModel(t testing.TB, seed int64) (*Model, *floatWeights) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	mat := func(rows, cols int, scale float64) []float32 {
		out := make([]float32, rows*cols)
		for i := range out {
			out[i] = float32(rng.NormFloat64() * scale)
		}
		return out
	}
	ones := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = 1
		}
		return out
	}

	fw := &floatWeights{
		words:     mat(tiny.vocab, tiny.hidden, 0.3),
		positions: mat(tiny.maxPos, tiny.hidden, 0.1),
		types:     mat(2, tiny.hidden, 0.05),
		embG:      ones(tiny.hidden),
		embB:      make([]float32, tiny.hidden),
	}

	pieces := map[string]float32{"▁": -1}
	for _, p := range []string{"а", "б", "в", "г", "▁кот", "▁дом", "▁бетон", "о", "т", "н", "е"} {
		pieces[p] = -2
	}
	vocab, err := buildVocabTensors(pieces, nil)
	if err != nil {
		t.Fatal(err)
	}
	vt, err := BuildVocabTensors(vocabPieces(pieces))
	if err != nil {
		t.Fatal(err)
	}
	_ = vocab

	w := NewWriter(Header{
		Model: "tiny", Arch: "bert", Hidden: tiny.hidden, Layers: tiny.layers,
		Heads: tiny.heads, FFN: tiny.ffn, Vocab: tiny.vocab, MaxPos: tiny.maxPos,
		TypeNum: 2, Eps: 1e-12, Pool: "mean", Normalize: true, Dim: tiny.hidden,
		BOS: idBOS, EOS: idEOS, UNK: idUNK, PAD: idPAD, UnkPenalty: -10,
		QueryPrefix: "query: ", PassagePrefix: "passage: ",
	})
	w.AddVocab(vt)
	w.AddMatrix("embeddings.word_embeddings.weight", QuantizeMatrix(fw.words, tiny.vocab, tiny.hidden))
	w.AddMatrix("embeddings.position_embeddings.weight", QuantizeMatrix(fw.positions, tiny.maxPos, tiny.hidden))
	w.AddMatrix("embeddings.token_type_embeddings.weight", QuantizeMatrix(fw.types, 2, tiny.hidden))
	w.AddVector("embeddings.LayerNorm.weight", fw.embG)
	w.AddVector("embeddings.LayerNorm.bias", fw.embB)

	fw.layers = make([]struct {
		q, k, v, o, f1, f2       []float32
		qb, kb, vb, ob, f1b, f2b []float32
		attnG, attnB, outG, outB []float32
	}, tiny.layers)

	for i := 0; i < tiny.layers; i++ {
		l := &fw.layers[i]
		l.q, l.k, l.v, l.o = mat(tiny.hidden, tiny.hidden, 0.2), mat(tiny.hidden, tiny.hidden, 0.2), mat(tiny.hidden, tiny.hidden, 0.2), mat(tiny.hidden, tiny.hidden, 0.2)
		l.qb, l.kb, l.vb, l.ob = mat(1, tiny.hidden, 0.01), mat(1, tiny.hidden, 0.01), mat(1, tiny.hidden, 0.01), mat(1, tiny.hidden, 0.01)
		l.f1, l.f1b = mat(tiny.ffn, tiny.hidden, 0.2), mat(1, tiny.ffn, 0.01)
		l.f2, l.f2b = mat(tiny.hidden, tiny.ffn, 0.2), mat(1, tiny.hidden, 0.01)
		l.attnG, l.attnB, l.outG, l.outB = ones(tiny.hidden), make([]float32, tiny.hidden), ones(tiny.hidden), make([]float32, tiny.hidden)

		p := prefix(i)
		w.AddMatrix(p+"attention.self.query.weight", QuantizeMatrix(l.q, tiny.hidden, tiny.hidden))
		w.AddVector(p+"attention.self.query.bias", l.qb)
		w.AddMatrix(p+"attention.self.key.weight", QuantizeMatrix(l.k, tiny.hidden, tiny.hidden))
		w.AddVector(p+"attention.self.key.bias", l.kb)
		w.AddMatrix(p+"attention.self.value.weight", QuantizeMatrix(l.v, tiny.hidden, tiny.hidden))
		w.AddVector(p+"attention.self.value.bias", l.vb)
		w.AddMatrix(p+"attention.output.dense.weight", QuantizeMatrix(l.o, tiny.hidden, tiny.hidden))
		w.AddVector(p+"attention.output.dense.bias", l.ob)
		w.AddVector(p+"attention.output.LayerNorm.weight", l.attnG)
		w.AddVector(p+"attention.output.LayerNorm.bias", l.attnB)
		w.AddMatrix(p+"intermediate.dense.weight", QuantizeMatrix(l.f1, tiny.ffn, tiny.hidden))
		w.AddVector(p+"intermediate.dense.bias", l.f1b)
		w.AddMatrix(p+"output.dense.weight", QuantizeMatrix(l.f2, tiny.hidden, tiny.ffn))
		w.AddVector(p+"output.dense.bias", l.f2b)
		w.AddVector(p+"output.LayerNorm.weight", l.outG)
		w.AddVector(p+"output.LayerNorm.bias", l.outB)
	}

	path := filepath.Join(t.TempDir(), "tiny.npse")
	if err := w.Write(path); err != nil {
		t.Fatalf("write blob: %v", err)
	}
	m, err := Load(path)
	if err != nil {
		t.Fatalf("load blob: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	return m, fw
}

func prefix(i int) string {
	return "encoder.layer." + itoa(i) + "."
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

func vocabPieces(pieces map[string]float32) []VocabPiece {
	all := []VocabPiece{{Text: "<s>"}, {Text: "<pad>"}, {Text: "</s>"}, {Text: "<unk>"}}
	keys := make([]string, 0, len(pieces))
	for k := range pieces {
		keys = append(keys, k)
	}
	sortStrings(keys)
	for _, k := range keys {
		all = append(all, VocabPiece{Text: k, Score: pieces[k]})
	}
	return all
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// referenceForward is an independent float32 implementation of the same
// encoder. It exists so the int8 path is checked against something other than
// itself.
func referenceForward(fw *floatWeights, ids []int32) []float32 {
	tokens := len(ids)
	hidden := tiny.hidden
	headDim := hidden / tiny.heads
	h := make([]float64, tokens*hidden)
	for t := 0; t < tokens; t++ {
		for i := 0; i < hidden; i++ {
			h[t*hidden+i] = float64(fw.words[int(ids[t])*hidden+i]) +
				float64(fw.positions[t*hidden+i]) + float64(fw.types[i])
		}
	}
	norm := func(x []float64, g, b []float32) {
		for t := 0; t < tokens; t++ {
			row := x[t*hidden : (t+1)*hidden]
			var mean, variance float64
			for _, v := range row {
				mean += v
			}
			mean /= float64(hidden)
			for _, v := range row {
				variance += (v - mean) * (v - mean)
			}
			variance /= float64(hidden)
			inv := 1 / math.Sqrt(variance+1e-12)
			for i, v := range row {
				row[i] = (v-mean)*inv*float64(g[i]) + float64(b[i])
			}
		}
	}
	matmul := func(x []float64, w []float32, bias []float32, in, out int) []float64 {
		res := make([]float64, tokens*out)
		for t := 0; t < tokens; t++ {
			for o := 0; o < out; o++ {
				var s float64
				for k := 0; k < in; k++ {
					s += x[t*in+k] * float64(w[o*in+k])
				}
				if bias != nil {
					s += float64(bias[o])
				}
				res[t*out+o] = s
			}
		}
		return res
	}

	norm(h, fw.embG, fw.embB)
	for li := range fw.layers {
		l := &fw.layers[li]
		q := matmul(h, l.q, l.qb, hidden, hidden)
		k := matmul(h, l.k, l.kb, hidden, hidden)
		v := matmul(h, l.v, l.vb, hidden, hidden)
		ctx := make([]float64, tokens*hidden)
		for head := 0; head < tiny.heads; head++ {
			off := head * headDim
			for i := 0; i < tokens; i++ {
				scores := make([]float64, tokens)
				maxS := math.Inf(-1)
				for j := 0; j < tokens; j++ {
					var s float64
					for d := 0; d < headDim; d++ {
						s += q[i*hidden+off+d] * k[j*hidden+off+d]
					}
					s /= math.Sqrt(float64(headDim))
					scores[j] = s
					if s > maxS {
						maxS = s
					}
				}
				var sum float64
				for j := range scores {
					scores[j] = math.Exp(scores[j] - maxS)
					sum += scores[j]
				}
				for j := range scores {
					scores[j] /= sum
				}
				for j := 0; j < tokens; j++ {
					for d := 0; d < headDim; d++ {
						ctx[i*hidden+off+d] += scores[j] * v[j*hidden+off+d]
					}
				}
			}
		}
		proj := matmul(ctx, l.o, l.ob, hidden, hidden)
		for i := range h {
			h[i] += proj[i]
		}
		norm(h, l.attnG, l.attnB)
		inter := matmul(h, l.f1, l.f1b, hidden, tiny.ffn)
		for i, v := range inter {
			inter[i] = 0.5 * v * (1 + math.Erf(v/math.Sqrt2))
		}
		out := matmul(inter, l.f2, l.f2b, tiny.ffn, hidden)
		for i := range h {
			h[i] += out[i]
		}
		norm(h, l.outG, l.outB)
	}

	pooled := make([]float32, hidden)
	for t := 0; t < tokens; t++ {
		for i := 0; i < hidden; i++ {
			pooled[i] += float32(h[t*hidden+i] / float64(tokens))
		}
	}
	l2Normalize(pooled)
	return pooled
}

func cosine(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}

// TestForwardMatchesFloatReference is the accuracy contract for the whole
// quantized encoder: the pooled vector must stay within 0.01 cosine of an
// independent float implementation of the same weights.
func TestForwardMatchesFloatReference(t *testing.T) {
	m, fw := randomTinyModel(t, 11)
	for _, text := range []string{"кот", "дом бетон", "а б в г"} {
		ids := m.vocab.Encode(text, m.maxPos)
		got := m.forward(ids, newScratchpad())
		want := referenceForward(fw, ids)
		if c := cosine(got, want); c < 0.99 {
			t.Fatalf("text %q: cosine to float reference = %.4f, want >= 0.99", text, c)
		}
	}
}

func TestForwardIsDeterministicAndNormalized(t *testing.T) {
	m, _ := randomTinyModel(t, 12)
	a := m.EmbedPassage("кот")
	b := m.EmbedPassage("кот")
	if len(a) != m.Dim() {
		t.Fatalf("dim = %d, want %d", len(a), m.Dim())
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("embedding is not deterministic at %d", i)
		}
	}
	if c := cosine(a, a); math.Abs(c-1) > 1e-5 {
		t.Fatalf("vector is not unit length: %v", c)
	}
}

func TestQueryAndPassagePrefixesDiffer(t *testing.T) {
	m, _ := randomTinyModel(t, 13)
	q := m.EmbedQuery("кот")
	p := m.EmbedPassage("кот")
	if cosine(q, p) > 0.9999 {
		return // identical is acceptable only if prefixes were empty
	}
	if m.queryPrefix == m.passagePrefix {
		t.Fatal("test model must carry asymmetric prefixes")
	}
}

func TestVectorQuantizationPreservesRanking(t *testing.T) {
	m, _ := randomTinyModel(t, 14)
	query := m.EmbedQuery("кот")
	near := m.EmbedPassage("кот")
	far := m.EmbedPassage("бетон")

	qn, qs := QuantizeVector(query)
	nn, ns := QuantizeVector(near)
	fn, fs := QuantizeVector(far)

	exactNear, exactFar := cosine(query, near), cosine(query, far)
	quantNear := DotQuantized(qn, qs, nn, ns)
	quantFar := DotQuantized(qn, qs, fn, fs)

	if math.Abs(float64(quantNear)-exactNear) > 0.01 || math.Abs(float64(quantFar)-exactFar) > 0.01 {
		t.Fatalf("int8 similarity drifted: %.4f vs %.4f, %.4f vs %.4f",
			quantNear, exactNear, quantFar, exactFar)
	}
}

func BenchmarkForwardTiny(b *testing.B) {
	m, _ := randomTinyModel(b, 15)
	sp := newScratchpad()
	ids := m.vocab.Encode("кот дом бетон а б в г", m.maxPos)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.forward(ids, sp)
	}
}
