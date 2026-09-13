package embed

import (
	"encoding/binary"
	"fmt"
	"sort"
	"unicode/utf8"
)

// VocabPiece is one entry of a unigram vocabulary, indexed by token id.
type VocabPiece struct {
	Text  string
	Score float32
}

// VocabTensors is the serialized, lookup-ready form of a vocabulary.
type VocabTensors struct {
	Blob     []byte
	Offsets  []uint32
	IDs      []uint32
	Scores   []float32
	MaxPiece int
}

// BuildVocabTensors sorts pieces lexicographically and packs them so the
// runtime can binary-search mapped bytes instead of building a hash map.
func BuildVocabTensors(pieces []VocabPiece) (*VocabTensors, error) {
	if len(pieces) == 0 {
		return nil, fmt.Errorf("embed: empty vocabulary")
	}
	order := make([]int, len(pieces))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return pieces[order[a]].Text < pieces[order[b]].Text
	})

	out := &VocabTensors{
		Offsets: make([]uint32, 0, len(pieces)+1),
		IDs:     make([]uint32, 0, len(pieces)),
		Scores:  make([]float32, 0, len(pieces)),
	}
	out.Offsets = append(out.Offsets, 0)
	for _, idx := range order {
		p := pieces[idx]
		if len(p.Text) > out.MaxPiece {
			out.MaxPiece = len(p.Text)
		}
		out.Blob = append(out.Blob, p.Text...)
		out.Offsets = append(out.Offsets, uint32(len(out.Blob)))
		out.IDs = append(out.IDs, uint32(idx))
		out.Scores = append(out.Scores, p.Score)
	}
	return out, nil
}

// Vocab builds a runtime vocabulary directly from packed tensors, used by the
// converter's self-check and by tests.
func (vt *VocabTensors) Vocab(bos, eos, unk int32, unkPenalty float64) *Vocab {
	return &Vocab{
		blob: vt.Blob, offsets: vt.Offsets, ids: vt.IDs, scores: vt.Scores,
		maxPiece: vt.MaxPiece, unkID: unk, unkPenalty: unkPenalty, bos: bos, eos: eos,
	}
}

// U32Bytes serializes a uint32 slice little-endian for the blob writer.
func U32Bytes(v []uint32) []byte {
	out := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(out[i*4:], x)
	}
	return out
}

// F32Bytes serializes a float32 slice little-endian.
func F32Bytes(v []float32) []byte {
	out := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(out[i*4:], float32bits(x))
	}
	return out
}

// I8Bytes reinterprets an int8 slice as bytes.
func I8Bytes(v []int8) []byte {
	out := make([]byte, len(v))
	for i, x := range v {
		out[i] = byte(x)
	}
	return out
}

// reservedIDs are the special token ids XLM-RoBERTa uses.
const (
	idBOS = 0
	idPAD = 1
	idEOS = 2
	idUNK = 3
)

// buildVocabTensors is the test and self-check helper: it lays out the four
// special tokens at their XLM-RoBERTa ids and appends the given pieces.
func buildVocabTensors(pieces map[string]float32, _ map[string]int32) (*Vocab, error) {
	names := make([]string, 0, len(pieces))
	for p := range pieces {
		if !utf8.ValidString(p) {
			return nil, fmt.Errorf("embed: piece %q is not valid UTF-8", p)
		}
		names = append(names, p)
	}
	sort.Strings(names)

	all := []VocabPiece{
		{Text: "<s>"}, {Text: "<pad>"}, {Text: "</s>"}, {Text: "<unk>"},
	}
	for _, n := range names {
		all = append(all, VocabPiece{Text: n, Score: pieces[n]})
	}
	vt, err := BuildVocabTensors(all)
	if err != nil {
		return nil, err
	}
	return vt.Vocab(idBOS, idEOS, idUNK, -10), nil
}
