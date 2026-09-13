package embed

import (
	"fmt"
	"math"
	"sync"
)

// Scratch buffer keys.
const (
	bufHidden = iota
	bufQ
	bufK
	bufV
	bufCtx
	bufProj
	bufFFN
	bufScores
)

// layerWeights holds one transformer block.
type layerWeights struct {
	query, key, value, attnOut *QuantMatrix
	queryB, keyB, valueB       []float32
	attnOutB                   []float32
	attnNormG, attnNormB       []float32
	ffnIn, ffnOut              *QuantMatrix
	ffnInB, ffnOutB            []float32
	outNormG, outNormB         []float32
}

// Model is a loaded embedding model: weights stay mapped, so two processes
// share one copy and start-up costs nothing but page faults.
type Model struct {
	blob  *Blob
	vocab *Vocab

	words, positions, types *QuantMatrix
	embNormG, embNormB      []float32
	layers                  []layerWeights

	hidden, heads, headDim, ffn int
	maxPos                      int
	eps                         float64
	normalize                   bool
	dim                         int

	queryPrefix, passagePrefix string
	name                       string

	pool sync.Pool
}

// Load maps a weight blob and wires up the tensors.
func Load(path string) (*Model, error) {
	blob, err := OpenBlob(path)
	if err != nil {
		return nil, err
	}
	m, err := fromBlob(blob)
	if err != nil {
		blob.Close()
		return nil, err
	}
	return m, nil
}

func fromBlob(blob *Blob) (*Model, error) {
	h := blob.H
	if h.Arch != "bert" {
		return nil, fmt.Errorf("embed: architecture %q is not supported", h.Arch)
	}
	if h.Pool != "mean" {
		return nil, fmt.Errorf("embed: pooling %q is not supported", h.Pool)
	}
	if h.Heads == 0 || h.Hidden%h.Heads != 0 {
		return nil, fmt.Errorf("embed: hidden %d does not split into %d heads", h.Hidden, h.Heads)
	}
	vocab, err := LoadVocab(blob)
	if err != nil {
		return nil, err
	}
	m := &Model{
		blob: blob, vocab: vocab,
		hidden: h.Hidden, heads: h.Heads, headDim: h.Hidden / h.Heads, ffn: h.FFN,
		maxPos: h.MaxPos, eps: h.Eps, normalize: h.Normalize, dim: h.Dim,
		queryPrefix: h.QueryPrefix, passagePrefix: h.PassagePrefix, name: h.Model,
	}
	if m.eps == 0 {
		m.eps = 1e-12
	}
	if m.dim == 0 {
		m.dim = h.Hidden
	}

	load := func(name string) *QuantMatrix {
		if err != nil {
			return nil
		}
		var mat *QuantMatrix
		mat, err = blob.Matrix(name)
		return mat
	}
	vec := func(name string, n int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = blob.Vector(name, n)
		return v
	}

	m.words = load("embeddings.word_embeddings.weight")
	m.positions = load("embeddings.position_embeddings.weight")
	m.types = load("embeddings.token_type_embeddings.weight")
	m.embNormG = vec("embeddings.LayerNorm.weight", h.Hidden)
	m.embNormB = vec("embeddings.LayerNorm.bias", h.Hidden)

	m.layers = make([]layerWeights, h.Layers)
	for i := 0; i < h.Layers; i++ {
		p := fmt.Sprintf("encoder.layer.%d.", i)
		l := &m.layers[i]
		l.query = load(p + "attention.self.query.weight")
		l.queryB = vec(p+"attention.self.query.bias", h.Hidden)
		l.key = load(p + "attention.self.key.weight")
		l.keyB = vec(p+"attention.self.key.bias", h.Hidden)
		l.value = load(p + "attention.self.value.weight")
		l.valueB = vec(p+"attention.self.value.bias", h.Hidden)
		l.attnOut = load(p + "attention.output.dense.weight")
		l.attnOutB = vec(p+"attention.output.dense.bias", h.Hidden)
		l.attnNormG = vec(p+"attention.output.LayerNorm.weight", h.Hidden)
		l.attnNormB = vec(p+"attention.output.LayerNorm.bias", h.Hidden)
		l.ffnIn = load(p + "intermediate.dense.weight")
		l.ffnInB = vec(p+"intermediate.dense.bias", h.FFN)
		l.ffnOut = load(p + "output.dense.weight")
		l.ffnOutB = vec(p+"output.dense.bias", h.Hidden)
		l.outNormG = vec(p+"output.LayerNorm.weight", h.Hidden)
		l.outNormB = vec(p+"output.LayerNorm.bias", h.Hidden)
	}
	if err != nil {
		return nil, err
	}
	m.pool.New = func() any { return newScratchpad() }
	return m, nil
}

// Close releases the mapped weights.
func (m *Model) Close() error { return m.blob.Close() }

// Name is the model identifier stored with every vector so a model swap is
// detectable instead of silently mixing incompatible embeddings.
func (m *Model) Name() string { return m.name }

// Dim is the embedding width.
func (m *Model) Dim() int { return m.dim }

// MaxTokens is the longest sequence the position table supports.
func (m *Model) MaxTokens() int { return m.maxPos }

// Tokenize exposes the model's own tokenizer. It exists so the segmentation can
// be compared against the reference implementation the checkpoint was trained
// with: the normaliser here is NFKC rather than SentencePiece's precompiled
// charsmap, and that difference has to be measurable, not assumed.
func (m *Model) Tokenize(text string) []int32 {
	return m.vocab.Encode(text, m.maxPos)
}

// EmbedQuery embeds a search query, applying the model's query prefix.
func (m *Model) EmbedQuery(text string) []float32 {
	return m.embed(m.queryPrefix + text)
}

// EmbedPassage embeds an indexed chunk, applying the model's passage prefix.
// E5 models are trained with asymmetric prefixes; dropping them costs several
// points of retrieval quality.
func (m *Model) EmbedPassage(text string) []float32 {
	return m.embed(m.passagePrefix + text)
}

func (m *Model) embed(text string) []float32 {
	ids := m.vocab.Encode(text, m.maxPos)
	sp := m.pool.Get().(*scratchpad)
	defer m.pool.Put(sp)
	return m.forward(ids, sp)
}

// forward runs the encoder over one sequence and returns the pooled vector.
func (m *Model) forward(ids []int32, sp *scratchpad) []float32 {
	tokens := len(ids)
	hidden := m.hidden
	h := sp.buf(bufHidden, tokens*hidden)

	// Embedding lookup: word + position + token type, dequantised on the fly.
	for t := 0; t < tokens; t++ {
		row := h[t*hidden : (t+1)*hidden]
		id := int(ids[t])
		if id < 0 || id >= m.words.Rows {
			id = int(m.vocab.unkID)
		}
		wScale := m.words.Scales[id]
		wRow := m.words.Data[id*hidden : (id+1)*hidden]
		pScale := m.positions.Scales[t]
		pRow := m.positions.Data[t*hidden : (t+1)*hidden]
		tScale := m.types.Scales[0]
		tRow := m.types.Data[0:hidden]
		for i := range row {
			row[i] = float32(wRow[i])*wScale + float32(pRow[i])*pScale + float32(tRow[i])*tScale
		}
	}
	layerNorm(h, tokens, hidden, m.embNormG, m.embNormB, m.eps)

	q := sp.buf(bufQ, tokens*hidden)
	k := sp.buf(bufK, tokens*hidden)
	v := sp.buf(bufV, tokens*hidden)
	ctx := sp.buf(bufCtx, tokens*hidden)
	proj := sp.buf(bufProj, tokens*hidden)
	ffn := sp.buf(bufFFN, tokens*m.ffn)
	scores := sp.buf(bufScores, tokens*tokens)
	scale := float32(1 / math.Sqrt(float64(m.headDim)))

	for li := range m.layers {
		l := &m.layers[li]
		linear(q, h, tokens, l.query, l.queryB, sp)
		linear(k, h, tokens, l.key, l.keyB, sp)
		linear(v, h, tokens, l.value, l.valueB, sp)

		for head := 0; head < m.heads; head++ {
			off := head * m.headDim
			for i := 0; i < tokens; i++ {
				qi := q[i*hidden+off : i*hidden+off+m.headDim]
				row := scores[i*tokens : (i+1)*tokens]
				for j := 0; j < tokens; j++ {
					kj := k[j*hidden+off : j*hidden+off+m.headDim]
					var s float32
					for d := range qi {
						s += qi[d] * kj[d]
					}
					row[j] = s * scale
				}
				softmaxRow(row)
			}
			for i := 0; i < tokens; i++ {
				out := ctx[i*hidden+off : i*hidden+off+m.headDim]
				for d := range out {
					out[d] = 0
				}
				row := scores[i*tokens : (i+1)*tokens]
				for j, p := range row {
					if p == 0 {
						continue
					}
					vj := v[j*hidden+off : j*hidden+off+m.headDim]
					for d := range out {
						out[d] += p * vj[d]
					}
				}
			}
		}

		linear(proj, ctx, tokens, l.attnOut, l.attnOutB, sp)
		addInPlace(h, proj)
		layerNorm(h, tokens, hidden, l.attnNormG, l.attnNormB, m.eps)

		linear(ffn, h, tokens, l.ffnIn, l.ffnInB, sp)
		gelu(ffn)
		linear(proj, ffn, tokens, l.ffnOut, l.ffnOutB, sp)
		addInPlace(h, proj)
		layerNorm(h, tokens, hidden, l.outNormG, l.outNormB, m.eps)
	}

	// Mean pooling over every real token. Single sequences are never padded,
	// so the mask is implicit.
	out := make([]float32, hidden)
	for t := 0; t < tokens; t++ {
		row := h[t*hidden : (t+1)*hidden]
		for i, x := range row {
			out[i] += x
		}
	}
	inv := 1 / float32(tokens)
	for i := range out {
		out[i] *= inv
	}
	if m.dim < hidden {
		out = out[:m.dim]
	}
	if m.normalize {
		l2Normalize(out)
	}
	return out
}
