// Package embed is a self-contained text embedding engine: an XLM-RoBERTa
// unigram tokenizer, a BERT encoder forward pass and int8 kernels, with no cgo
// and no ONNX runtime. Everything the model needs lives in one memory-mapped
// blob so several processes on a Raspberry Pi share the same physical pages.
package embed

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"unsafe"
)

// Magic identifies a weight blob; the trailing digits are the format version.
const Magic = "NPSE0002"

// Alignment keeps every tensor 64-byte aligned so float and int32 views of the
// mapped bytes are safe on every architecture the binary targets.
const Alignment = 64

// TensorInfo describes one tensor inside the blob.
type TensorInfo struct {
	DType  string `json:"dtype"` // "f32", "i8", "u32", "i32", "u8"
	Shape  []int  `json:"shape"`
	Offset int64  `json:"offset"`
	Bytes  int64  `json:"bytes"`
	// Scales points at a per-row f32 scale tensor for i8 weights.
	Scales string `json:"scales,omitempty"`
}

// Header is the JSON preamble of a blob.
type Header struct {
	Format  string  `json:"format"`
	Model   string  `json:"model"`
	Arch    string  `json:"arch"`
	Hidden  int     `json:"hidden"`
	Layers  int     `json:"layers"`
	Heads   int     `json:"heads"`
	FFN     int     `json:"intermediate"`
	Vocab   int     `json:"vocab"`
	MaxPos  int     `json:"max_pos"`
	TypeNum int     `json:"type_vocab"`
	Eps     float64 `json:"layer_norm_eps"`

	Pool      string `json:"pool"`      // only "mean" is implemented
	Normalize bool   `json:"normalize"` // L2-normalise the pooled vector
	Dim       int    `json:"dim"`       // output dimensionality

	QueryPrefix   string `json:"query_prefix"`
	PassagePrefix string `json:"passage_prefix"`

	BOS int `json:"bos_id"`
	EOS int `json:"eos_id"`
	UNK int `json:"unk_id"`
	PAD int `json:"pad_id"`

	MaxPieceBytes int     `json:"max_piece_bytes"`
	UnkPenalty    float64 `json:"unk_penalty"`

	// AddedTokens maps the literal strings the reference tokenizer matches
	// before normalization ("<mask>", "<unk>", ...) to their ids. Absent in
	// blobs written before the table existed; the tokenizer then simply never
	// matches one, which is the old behaviour.
	AddedTokens map[string]int `json:"added_tokens,omitempty"`

	Tensors map[string]TensorInfo `json:"tensors"`
}

// Blob is a mapped weight file.
type Blob struct {
	H     Header
	data  []byte
	unmap func() error
}

// OpenBlob maps a weight blob read-only.
func OpenBlob(path string) (*Blob, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() < 16 {
		return nil, fmt.Errorf("embed: %s is too small to be a weight blob", path)
	}
	data, unmap, err := mapFile(f, int(st.Size()))
	if err != nil {
		return nil, err
	}
	b := &Blob{data: data, unmap: unmap}
	if string(data[:len(Magic)]) != Magic {
		b.Close()
		return nil, fmt.Errorf("embed: %s is not a %s blob", path, Magic)
	}
	headerLen := binary.LittleEndian.Uint32(data[len(Magic):])
	start := int64(len(Magic) + 4)
	if int64(headerLen)+start > st.Size() {
		b.Close()
		return nil, fmt.Errorf("embed: header length %d overruns the file", headerLen)
	}
	if err := json.Unmarshal(data[start:start+int64(headerLen)], &b.H); err != nil {
		b.Close()
		return nil, fmt.Errorf("embed: parse header: %w", err)
	}
	return b, nil
}

// Close unmaps the blob.
func (b *Blob) Close() error {
	if b.unmap == nil {
		return nil
	}
	err := b.unmap()
	b.unmap, b.data = nil, nil
	return err
}

// Size is the mapped size in bytes.
func (b *Blob) Size() int { return len(b.data) }

func (b *Blob) raw(name string) ([]byte, TensorInfo, error) {
	info, ok := b.H.Tensors[name]
	if !ok {
		return nil, info, fmt.Errorf("embed: tensor %q missing from blob", name)
	}
	if info.Offset < 0 || info.Offset+info.Bytes > int64(len(b.data)) {
		return nil, info, fmt.Errorf("embed: tensor %q out of range", name)
	}
	return b.data[info.Offset : info.Offset+info.Bytes], info, nil
}

// F32 returns a float32 view of a tensor without copying.
func (b *Blob) F32(name string) ([]float32, TensorInfo, error) {
	raw, info, err := b.raw(name)
	if err != nil {
		return nil, info, err
	}
	if info.DType != "f32" {
		return nil, info, fmt.Errorf("embed: tensor %q is %s, want f32", name, info.DType)
	}
	return bytesAsF32(raw), info, nil
}

// I8 returns an int8 view of a tensor without copying.
func (b *Blob) I8(name string) ([]int8, TensorInfo, error) {
	raw, info, err := b.raw(name)
	if err != nil {
		return nil, info, err
	}
	if info.DType != "i8" {
		return nil, info, fmt.Errorf("embed: tensor %q is %s, want i8", name, info.DType)
	}
	return bytesAsI8(raw), info, nil
}

// U32 returns a uint32 view of a tensor without copying.
func (b *Blob) U32(name string) ([]uint32, TensorInfo, error) {
	raw, info, err := b.raw(name)
	if err != nil {
		return nil, info, err
	}
	if info.DType != "u32" {
		return nil, info, fmt.Errorf("embed: tensor %q is %s, want u32", name, info.DType)
	}
	return bytesAsU32(raw), info, nil
}

// Bytes returns a raw byte tensor, used for the packed vocabulary strings.
func (b *Blob) Bytes(name string) ([]byte, error) {
	raw, _, err := b.raw(name)
	return raw, err
}

// QuantMatrix is an int8 weight matrix with one scale per output row.
type QuantMatrix struct {
	Rows, Cols int
	Data       []int8    // Rows*Cols, row-major
	Scales     []float32 // len == Rows
}

// Matrix loads an int8 matrix with its per-row scales.
func (b *Blob) Matrix(name string) (*QuantMatrix, error) {
	data, info, err := b.I8(name)
	if err != nil {
		return nil, err
	}
	if len(info.Shape) != 2 {
		return nil, fmt.Errorf("embed: tensor %q has shape %v, want 2 dims", name, info.Shape)
	}
	if info.Scales == "" {
		return nil, fmt.Errorf("embed: tensor %q has no scales", name)
	}
	scales, _, err := b.F32(info.Scales)
	if err != nil {
		return nil, err
	}
	rows, cols := info.Shape[0], info.Shape[1]
	if len(data) != rows*cols {
		return nil, fmt.Errorf("embed: tensor %q holds %d values, want %d", name, len(data), rows*cols)
	}
	if len(scales) != rows {
		return nil, fmt.Errorf("embed: tensor %q has %d scales, want %d", name, len(scales), rows)
	}
	return &QuantMatrix{Rows: rows, Cols: cols, Data: data, Scales: scales}, nil
}

// Vector loads a float32 bias or LayerNorm parameter.
func (b *Blob) Vector(name string, want int) ([]float32, error) {
	v, _, err := b.F32(name)
	if err != nil {
		return nil, err
	}
	if len(v) != want {
		return nil, fmt.Errorf("embed: tensor %q holds %d values, want %d", name, len(v), want)
	}
	return v, nil
}

func bytesAsF32(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func bytesAsI8(b []byte) []int8 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*int8)(unsafe.Pointer(&b[0])), len(b))
}

func bytesAsU32(b []byte) []uint32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// nearestInt rounds half away from zero, matching the reference quantizers.
func nearestInt(v float32) int32 {
	if v >= 0 {
		return int32(math.Floor(float64(v) + 0.5))
	}
	return -int32(math.Floor(float64(-v) + 0.5))
}
