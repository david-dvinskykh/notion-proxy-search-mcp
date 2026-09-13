package embed

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// buildFakeCheckpoint writes a minimal but structurally complete BERT
// checkpoint so the converter is exercised end to end without a download.
func buildFakeCheckpoint(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	const hidden, layers, heads, ffn, maxPos, types = 8, 1, 2, 16, 12, 2

	cfg := map[string]any{
		"architectures": []string{"BertModel"}, "model_type": "bert", "hidden_act": "gelu",
		"hidden_size": hidden, "num_hidden_layers": layers, "num_attention_heads": heads,
		"intermediate_size": ffn, "max_position_embeddings": maxPos, "type_vocab_size": types,
		"layer_norm_eps": 1e-12, "vocab_size": 6,
	}
	cfgBytes, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgBytes, 0o600); err != nil {
		t.Fatal(err)
	}

	tok := map[string]any{
		"model": map[string]any{
			"type":   "Unigram",
			"unk_id": 3,
			"vocab": [][]any{
				{"<s>", 0.0}, {"<pad>", 0.0}, {"</s>", 0.0}, {"<unk>", 0.0},
				{"▁", -2.5}, {"▁проверка", -1.0},
			},
		},
		"added_tokens": []map[string]any{
			{"id": 0, "content": "<s>"}, {"id": 1, "content": "<pad>"},
			{"id": 2, "content": "</s>"}, {"id": 3, "content": "<unk>"},
		},
	}
	tokBytes, _ := json.Marshal(tok)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), tokBytes, 0o600); err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(5))
	tensors := map[string][]float32{}
	shapes := map[string][]int{}
	add := func(name string, dims ...int) {
		n := 1
		for _, d := range dims {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * 0.1)
		}
		tensors[name], shapes[name] = v, dims
	}
	ones := func(name string, n int) {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1
		}
		tensors[name], shapes[name] = v, []int{n}
	}
	add("embeddings.word_embeddings.weight", 6, hidden)
	add("embeddings.position_embeddings.weight", maxPos, hidden)
	add("embeddings.token_type_embeddings.weight", types, hidden)
	ones("embeddings.LayerNorm.weight", hidden)
	add("embeddings.LayerNorm.bias", hidden)
	p := "encoder.layer.0."
	for _, part := range []string{"query", "key", "value"} {
		add(p+"attention.self."+part+".weight", hidden, hidden)
		add(p+"attention.self."+part+".bias", hidden)
	}
	add(p+"attention.output.dense.weight", hidden, hidden)
	add(p+"attention.output.dense.bias", hidden)
	ones(p+"attention.output.LayerNorm.weight", hidden)
	add(p+"attention.output.LayerNorm.bias", hidden)
	add(p+"intermediate.dense.weight", ffn, hidden)
	add(p+"intermediate.dense.bias", ffn)
	add(p+"output.dense.weight", hidden, ffn)
	add(p+"output.dense.bias", hidden)
	ones(p+"output.LayerNorm.weight", hidden)
	add(p+"output.LayerNorm.bias", hidden)

	type entry struct {
		DType   string   `json:"dtype"`
		Shape   []int    `json:"shape"`
		Offsets [2]int64 `json:"data_offsets"`
	}
	index := map[string]entry{}
	var body []byte
	for name, values := range tensors {
		start := int64(len(body))
		for _, v := range values {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			body = append(body, buf[:]...)
		}
		index[name] = entry{DType: "F32", Shape: shapes[name], Offsets: [2]int64{start, int64(len(body))}}
	}
	header, _ := json.Marshal(index)
	var out []byte
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(header)))
	out = append(out, lenBuf[:]...)
	out = append(out, header...)
	out = append(out, body...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), out, 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestConvertProducesLoadableBlob(t *testing.T) {
	dir := buildFakeCheckpoint(t)
	out := filepath.Join(t.TempDir(), "model.npse")
	if err := Convert(ConvertOptions{ModelDir: dir, Out: out, ModelName: "fake-e5"}); err != nil {
		t.Fatalf("convert: %v", err)
	}

	m, err := Load(out)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if m.Name() != "fake-e5" {
		t.Fatalf("model name = %q", m.Name())
	}
	v := m.EmbedPassage("проверка")
	if len(v) != m.Dim() {
		t.Fatalf("dim = %d, want %d", len(v), m.Dim())
	}
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	if math.Abs(norm-1) > 1e-4 {
		t.Fatalf("vector norm² = %v", norm)
	}
}

func TestConvertRejectsUnsupportedArchitecture(t *testing.T) {
	dir := buildFakeCheckpoint(t)
	cfg, _ := json.Marshal(map[string]any{"model_type": "roberta", "hidden_act": "gelu"})
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfg, 0o600); err != nil {
		t.Fatal(err)
	}
	err := Convert(ConvertOptions{ModelDir: dir, Out: filepath.Join(t.TempDir(), "x.npse")})
	if err == nil {
		t.Fatal("expected an error for a non-BERT checkpoint")
	}
}

func TestLoadUnigramVocabReadsScoresAndSpecials(t *testing.T) {
	dir := buildFakeCheckpoint(t)
	pieces, sp, err := LoadUnigramVocab(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(pieces) != 6 {
		t.Fatalf("got %d pieces", len(pieces))
	}
	if pieces[5].Text != "▁проверка" || pieces[5].Score != -1 {
		t.Fatalf("piece 5 = %+v", pieces[5])
	}
	if sp.BOS != 0 || sp.EOS != 2 || sp.Unk != 3 || sp.Pad != 1 {
		t.Fatalf("specials = %+v", sp)
	}
	if sp.UnkPenalty >= -2.5 {
		t.Fatalf("unk penalty %v must be worse than the worst piece score", sp.UnkPenalty)
	}
}
