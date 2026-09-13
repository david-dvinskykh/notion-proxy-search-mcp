package embed

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// Writer assembles a weight blob: a JSON header followed by 64-byte aligned
// tensor payloads.
type Writer struct {
	header   Header
	nameList []string
	chunks   [][]byte
}

// NewWriter starts a blob with the given header; tensors are added after.
func NewWriter(h Header) *Writer {
	h.Format = Magic
	if h.Tensors == nil {
		h.Tensors = map[string]TensorInfo{}
	}
	return &Writer{header: h}
}

// AddRaw appends a tensor payload with an explicit dtype and shape.
func (w *Writer) AddRaw(name, dtype string, shape []int, payload []byte, scales string) {
	w.header.Tensors[name] = TensorInfo{
		DType: dtype, Shape: shape, Bytes: int64(len(payload)), Scales: scales,
	}
	w.nameList = append(w.nameList, name)
	w.chunks = append(w.chunks, payload)
}

// AddMatrix appends a quantized matrix together with its scale vector.
func (w *Writer) AddMatrix(name string, m *QuantMatrix) {
	scaleName := name + ".scales"
	w.AddRaw(name, "i8", []int{m.Rows, m.Cols}, I8Bytes(m.Data), scaleName)
	w.AddRaw(scaleName, "f32", []int{m.Rows}, F32Bytes(m.Scales), "")
}

// AddVector appends a float32 vector.
func (w *Writer) AddVector(name string, v []float32) {
	w.AddRaw(name, "f32", []int{len(v)}, F32Bytes(v), "")
}

// AddVocab appends the packed vocabulary tensors.
func (w *Writer) AddVocab(vt *VocabTensors) {
	w.AddRaw("vocab.blob", "u8", []int{len(vt.Blob)}, vt.Blob, "")
	w.AddRaw("vocab.offsets", "u32", []int{len(vt.Offsets)}, U32Bytes(vt.Offsets), "")
	w.AddRaw("vocab.ids", "u32", []int{len(vt.IDs)}, U32Bytes(vt.IDs), "")
	w.AddRaw("vocab.scores", "f32", []int{len(vt.Scores)}, F32Bytes(vt.Scores), "")
	w.header.MaxPieceBytes = vt.MaxPiece
}

// Write serializes the blob to path.
func (w *Writer) Write(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := w.Serialize(f); err != nil {
		return err
	}
	return f.Sync()
}

// Serialize writes the blob to out.

func (w *Writer) Serialize(out io.Writer) error {
	// Two passes: assign offsets against a provisional header size, then grow
	// the padding so the real header fits. The header only changes in length
	// when offsets change, so one correction round converges.
	headerLen := 0
	for attempt := 0; attempt < 8; attempt++ {
		base := int64(len(Magic) + 4 + headerLen)
		base = align(base)
		cursor := base
		for i, name := range w.nameList {
			info := w.header.Tensors[name]
			info.Offset = cursor
			w.header.Tensors[name] = info
			cursor = align(cursor + int64(len(w.chunks[i])))
		}
		encoded, err := json.Marshal(w.header)
		if err != nil {
			return err
		}
		if len(encoded) == headerLen {
			return w.emit(out, encoded, base)
		}
		headerLen = len(encoded)
	}
	return fmt.Errorf("embed: header size did not converge")
}

func (w *Writer) emit(out io.Writer, header []byte, base int64) error {
	if _, err := out.Write([]byte(Magic)); err != nil {
		return err
	}
	var lenBuf [4]byte
	binary.LittleEndian.PutUint32(lenBuf[:], uint32(len(header)))
	if _, err := out.Write(lenBuf[:]); err != nil {
		return err
	}
	if _, err := out.Write(header); err != nil {
		return err
	}
	written := int64(len(Magic) + 4 + len(header))
	if err := pad(out, base-written); err != nil {
		return err
	}
	written = base
	for i, name := range w.nameList {
		info := w.header.Tensors[name]
		if info.Offset != written {
			return fmt.Errorf("embed: tensor %s offset drift: header %d, stream %d", name, info.Offset, written)
		}
		if _, err := out.Write(w.chunks[i]); err != nil {
			return err
		}
		written += int64(len(w.chunks[i]))
		next := align(written)
		if err := pad(out, next-written); err != nil {
			return err
		}
		written = next
	}
	return nil
}

func align(n int64) int64 {
	if rem := n % Alignment; rem != 0 {
		return n + Alignment - rem
	}
	return n
}

var zeros [Alignment]byte

func pad(out io.Writer, n int64) error {
	for n > 0 {
		step := n
		if step > int64(len(zeros)) {
			step = int64(len(zeros))
		}
		if _, err := out.Write(zeros[:step]); err != nil {
			return err
		}
		n -= step
	}
	return nil
}
