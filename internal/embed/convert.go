package embed

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// ConvertOptions describes a Hugging Face checkpoint to turn into a blob.
type ConvertOptions struct {
	ModelDir  string // holds config.json, tokenizer.json, model.safetensors
	Out       string
	ModelName string
	// QueryPrefix and PassagePrefix default to the E5 prefixes, which the
	// model was trained with; an empty pair disables them.
	QueryPrefix   string
	PassagePrefix string
	Dim           int // truncate the pooled vector (Matryoshka-style); 0 keeps all
	Logf          func(format string, args ...any)
}

// hfConfig is the subset of config.json the converter needs.
type hfConfig struct {
	Architectures     []string `json:"architectures"`
	HiddenSize        int      `json:"hidden_size"`
	NumHiddenLayers   int      `json:"num_hidden_layers"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	IntermediateSize  int      `json:"intermediate_size"`
	MaxPositions      int      `json:"max_position_embeddings"`
	TypeVocabSize     int      `json:"type_vocab_size"`
	LayerNormEps      float64  `json:"layer_norm_eps"`
	VocabSize         int      `json:"vocab_size"`
	HiddenAct         string   `json:"hidden_act"`
	ModelType         string   `json:"model_type"`
}

// Convert quantizes a checkpoint into a single blob the runtime maps.
func Convert(opts ConvertOptions) error {
	logf := opts.Logf
	if logf == nil {
		logf = func(string, ...any) {}
	}
	if opts.ModelDir == "" || opts.Out == "" {
		return fmt.Errorf("embed: convert needs a model directory and an output path")
	}

	cfgBytes, err := os.ReadFile(opts.ModelDir + "/config.json")
	if err != nil {
		return fmt.Errorf("embed: read config.json: %w", err)
	}
	var cfg hfConfig
	if err := json.Unmarshal(cfgBytes, &cfg); err != nil {
		return fmt.Errorf("embed: parse config.json: %w", err)
	}
	if cfg.ModelType != "bert" {
		return fmt.Errorf("embed: model_type %q is not supported (only bert)", cfg.ModelType)
	}
	if cfg.HiddenAct != "gelu" {
		return fmt.Errorf("embed: hidden_act %q is not supported (only gelu)", cfg.HiddenAct)
	}

	pieces, specials, err := LoadUnigramVocab(opts.ModelDir + "/tokenizer.json")
	if err != nil {
		return err
	}
	logf("vocabulary: %d pieces, unk=%d bos=%d eos=%d", len(pieces), specials.Unk, specials.BOS, specials.EOS)

	st, err := OpenSafetensors(opts.ModelDir + "/model.safetensors")
	if err != nil {
		return err
	}
	defer st.Close()

	prefix, err := detectPrefix(st)
	if err != nil {
		return err
	}
	if prefix != "" {
		logf("stripping checkpoint prefix %q", prefix)
	}

	dim := opts.Dim
	if dim == 0 || dim > cfg.HiddenSize {
		dim = cfg.HiddenSize
	}
	queryPrefix, passagePrefix := opts.QueryPrefix, opts.PassagePrefix
	if queryPrefix == "" && passagePrefix == "" {
		queryPrefix, passagePrefix = "query: ", "passage: "
	}
	eps := cfg.LayerNormEps
	if eps == 0 {
		eps = 1e-12
	}

	vt, err := BuildVocabTensors(pieces)
	if err != nil {
		return err
	}

	name := opts.ModelName
	if name == "" {
		name = "bert"
	}
	w := NewWriter(Header{
		Model: name, Arch: "bert",
		Hidden: cfg.HiddenSize, Layers: cfg.NumHiddenLayers, Heads: cfg.NumAttentionHeads,
		FFN: cfg.IntermediateSize, Vocab: len(pieces), MaxPos: cfg.MaxPositions,
		TypeNum: cfg.TypeVocabSize, Eps: eps,
		Pool: "mean", Normalize: true, Dim: dim,
		QueryPrefix: queryPrefix, PassagePrefix: passagePrefix,
		BOS: specials.BOS, EOS: specials.EOS, UNK: specials.Unk, PAD: specials.Pad,
		UnkPenalty: specials.UnkPenalty, AddedTokens: specials.Added,
	})
	w.AddVocab(vt)

	addMatrix := func(blobName, checkpointName string, rows, cols int) error {
		values, shape, err := st.Float32(prefix + checkpointName)
		if err != nil {
			return err
		}
		if len(shape) != 2 || shape[0] != rows || shape[1] != cols {
			return fmt.Errorf("embed: %s has shape %v, want [%d %d]", checkpointName, shape, rows, cols)
		}
		w.AddMatrix(blobName, QuantizeMatrix(values, rows, cols))
		return nil
	}
	// addEmbeddingTable is addMatrix with a relaxed row count. Checkpoints
	// routinely pad the word embedding table past the tokenizer's vocabulary —
	// multilingual-e5-small carries 250037 rows for 250002 pieces — because the
	// table gets rounded up for tensor alignment. Every row is kept: an added
	// token can legitimately sit above the piece count, and the runtime indexes
	// the table by token id.
	addEmbeddingTable := func(blobName, checkpointName string, minRows, cols int) (int, error) {
		values, shape, err := st.Float32(prefix + checkpointName)
		if err != nil {
			return 0, err
		}
		if len(shape) != 2 || shape[1] != cols {
			return 0, fmt.Errorf("embed: %s has shape %v, want [rows %d]", checkpointName, shape, cols)
		}
		if shape[0] < minRows {
			return 0, fmt.Errorf("embed: %s holds %d rows for %d vocabulary pieces", checkpointName, shape[0], minRows)
		}
		w.AddMatrix(blobName, QuantizeMatrix(values, shape[0], cols))
		return shape[0], nil
	}
	addVector := func(blobName, checkpointName string, n int) error {
		values, shape, err := st.Float32(prefix + checkpointName)
		if err != nil {
			return err
		}
		if len(shape) != 1 || shape[0] != n {
			return fmt.Errorf("embed: %s has shape %v, want [%d]", checkpointName, shape, n)
		}
		w.AddVector(blobName, values)
		return nil
	}

	h, ffn := cfg.HiddenSize, cfg.IntermediateSize
	tableRows, err := addEmbeddingTable("embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight", len(pieces), h)
	if err != nil {
		return err
	}
	if tableRows != len(pieces) {
		logf("word embedding table has %d rows for %d pieces: %d padding rows kept", tableRows, len(pieces), tableRows-len(pieces))
	}
	steps := []func() error{
		func() error {
			return addMatrix("embeddings.position_embeddings.weight", "embeddings.position_embeddings.weight", cfg.MaxPositions, h)
		},
		func() error {
			return addMatrix("embeddings.token_type_embeddings.weight", "embeddings.token_type_embeddings.weight", cfg.TypeVocabSize, h)
		},
		func() error { return addVector("embeddings.LayerNorm.weight", "embeddings.LayerNorm.weight", h) },
		func() error { return addVector("embeddings.LayerNorm.bias", "embeddings.LayerNorm.bias", h) },
	}
	for _, step := range steps {
		if err := step(); err != nil {
			return err
		}
	}

	for i := 0; i < cfg.NumHiddenLayers; i++ {
		p := fmt.Sprintf("encoder.layer.%d.", i)
		layerSteps := []func() error{
			func() error { return addMatrix(p+"attention.self.query.weight", p+"attention.self.query.weight", h, h) },
			func() error { return addVector(p+"attention.self.query.bias", p+"attention.self.query.bias", h) },
			func() error { return addMatrix(p+"attention.self.key.weight", p+"attention.self.key.weight", h, h) },
			func() error { return addVector(p+"attention.self.key.bias", p+"attention.self.key.bias", h) },
			func() error { return addMatrix(p+"attention.self.value.weight", p+"attention.self.value.weight", h, h) },
			func() error { return addVector(p+"attention.self.value.bias", p+"attention.self.value.bias", h) },
			func() error {
				return addMatrix(p+"attention.output.dense.weight", p+"attention.output.dense.weight", h, h)
			},
			func() error { return addVector(p+"attention.output.dense.bias", p+"attention.output.dense.bias", h) },
			func() error {
				return addVector(p+"attention.output.LayerNorm.weight", p+"attention.output.LayerNorm.weight", h)
			},
			func() error {
				return addVector(p+"attention.output.LayerNorm.bias", p+"attention.output.LayerNorm.bias", h)
			},
			func() error { return addMatrix(p+"intermediate.dense.weight", p+"intermediate.dense.weight", ffn, h) },
			func() error { return addVector(p+"intermediate.dense.bias", p+"intermediate.dense.bias", ffn) },
			func() error { return addMatrix(p+"output.dense.weight", p+"output.dense.weight", h, ffn) },
			func() error { return addVector(p+"output.dense.bias", p+"output.dense.bias", h) },
			func() error { return addVector(p+"output.LayerNorm.weight", p+"output.LayerNorm.weight", h) },
			func() error { return addVector(p+"output.LayerNorm.bias", p+"output.LayerNorm.bias", h) },
		}
		for _, step := range layerSteps {
			if err := step(); err != nil {
				return err
			}
		}
		logf("layer %d/%d quantized", i+1, cfg.NumHiddenLayers)
	}

	if err := w.Write(opts.Out); err != nil {
		return err
	}
	info, err := os.Stat(opts.Out)
	if err != nil {
		return err
	}
	logf("wrote %s (%.1f MiB)", opts.Out, float64(info.Size())/(1<<20))

	// Self-check: the blob must load and embed without error. A converter that
	// writes an unreadable blob is worse than one that fails loudly.
	m, err := Load(opts.Out)
	if err != nil {
		return fmt.Errorf("embed: self-check failed to load blob: %w", err)
	}
	defer m.Close()
	v := m.EmbedQuery("проверка")
	if len(v) != dim {
		return fmt.Errorf("embed: self-check returned %d dims, want %d", len(v), dim)
	}
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum < 0.99 || sum > 1.01 {
		return fmt.Errorf("embed: self-check vector is not unit length (norm² = %.4f)", sum)
	}
	logf("self-check passed: %d-dim unit vector", len(v))
	return nil
}

// detectPrefix finds the prefix the checkpoint uses for BERT tensors. Plain
// BertModel exports have none; sentence-transformers exports sometimes nest
// them under "bert." or "0.auto_model.".
func detectPrefix(st *Safetensors) (string, error) {
	const probe = "embeddings.word_embeddings.weight"
	for _, p := range []string{"", "bert.", "0.auto_model.", "model."} {
		if _, ok := st.Shape(p + probe); ok {
			return p, nil
		}
	}
	names := st.Names()
	if len(names) > 8 {
		names = names[:8]
	}
	return "", fmt.Errorf("embed: no %s tensor found; checkpoint holds e.g. %s", probe, strings.Join(names, ", "))
}

// Specials carries the ids of the reserved tokens.
type Specials struct {
	BOS, EOS, Unk, Pad int
	UnkPenalty         float64

	// Added holds every added token of the checkpoint by its literal content:
	// the reference tokenizer cuts the input on these strings before it
	// normalizes anything, so the Go tokenizer needs the same table.
	Added map[string]int
}

// LoadUnigramVocab reads the unigram vocabulary out of a tokenizer.json.
func LoadUnigramVocab(path string) ([]VocabPiece, Specials, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, Specials{}, fmt.Errorf("embed: read tokenizer.json: %w", err)
	}
	var doc struct {
		Model struct {
			Type  string            `json:"type"`
			Vocab []json.RawMessage `json:"vocab"`
			UnkID *int              `json:"unk_id"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, Specials{}, fmt.Errorf("embed: parse tokenizer.json: %w", err)
	}
	if doc.Model.Type != "Unigram" {
		return nil, Specials{}, fmt.Errorf("embed: tokenizer model %q is not Unigram", doc.Model.Type)
	}

	pieces := make([]VocabPiece, 0, len(doc.Model.Vocab))
	minScore := 0.0
	for i, entry := range doc.Model.Vocab {
		var pair []json.RawMessage
		if err := json.Unmarshal(entry, &pair); err != nil || len(pair) != 2 {
			return nil, Specials{}, fmt.Errorf("embed: vocab entry %d is not a [piece, score] pair", i)
		}
		var text string
		var score float64
		if err := json.Unmarshal(pair[0], &text); err != nil {
			return nil, Specials{}, fmt.Errorf("embed: vocab entry %d piece: %w", i, err)
		}
		if err := json.Unmarshal(pair[1], &score); err != nil {
			return nil, Specials{}, fmt.Errorf("embed: vocab entry %d score: %w", i, err)
		}
		if score < minScore {
			minScore = score
		}
		pieces = append(pieces, VocabPiece{Text: text, Score: float32(score)})
	}

	sp := Specials{BOS: idBOS, Pad: idPAD, EOS: idEOS, Unk: idUNK}
	if doc.Model.UnkID != nil {
		sp.Unk = *doc.Model.UnkID
	}
	byContent := map[string]int{}
	for _, a := range doc.AddedTokens {
		byContent[a.Content] = a.ID
	}
	sp.Added = byContent
	for content, target := range map[string]*int{"<s>": &sp.BOS, "</s>": &sp.EOS, "<pad>": &sp.Pad, "<unk>": &sp.Unk} {
		if id, ok := byContent[content]; ok {
			*target = id
		}
	}
	// An unknown rune must cost more than any real piece, otherwise Viterbi
	// prefers spelling words out of <unk>.
	sp.UnkPenalty = minScore - 10
	return pieces, sp, nil
}
