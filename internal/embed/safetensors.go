package embed

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Safetensors is a memory-mapped safetensors file.
type Safetensors struct {
	tensors map[string]stTensor
	data    []byte
	unmap   func() error
}

type stTensor struct {
	DType   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

// OpenSafetensors maps a safetensors file and parses its index.
func OpenSafetensors(path string) (*Safetensors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() < 8 {
		return nil, fmt.Errorf("embed: %s is too small for safetensors", path)
	}
	data, unmap, err := mapFile(f, int(st.Size()))
	if err != nil {
		return nil, err
	}
	headerLen := int64(binary.LittleEndian.Uint64(data[:8]))
	if headerLen <= 0 || 8+headerLen > st.Size() {
		_ = unmap()
		return nil, fmt.Errorf("embed: safetensors header length %d is out of range", headerLen)
	}
	var index map[string]json.RawMessage
	if err := json.Unmarshal(data[8:8+headerLen], &index); err != nil {
		_ = unmap()
		return nil, fmt.Errorf("embed: parse safetensors header: %w", err)
	}
	out := &Safetensors{tensors: map[string]stTensor{}, data: data[8+headerLen:], unmap: unmap}
	for name, raw := range index {
		if name == "__metadata__" {
			continue
		}
		var t stTensor
		if err := json.Unmarshal(raw, &t); err != nil {
			_ = unmap()
			return nil, fmt.Errorf("embed: tensor %q: %w", name, err)
		}
		out.tensors[name] = t
	}
	return out, nil
}

// Close unmaps the file.
func (s *Safetensors) Close() error {
	if s.unmap == nil {
		return nil
	}
	err := s.unmap()
	s.unmap = nil
	return err
}

// Names lists every tensor in the file.
func (s *Safetensors) Names() []string {
	out := make([]string, 0, len(s.tensors))
	for n := range s.tensors {
		out = append(out, n)
	}
	return out
}

// Shape returns a tensor's dimensions.
func (s *Safetensors) Shape(name string) ([]int, bool) {
	t, ok := s.tensors[name]
	if !ok {
		return nil, false
	}
	return t.Shape, true
}

// Float32 materialises a tensor as float32, converting from half precision
// where needed.
func (s *Safetensors) Float32(name string) ([]float32, []int, error) {
	t, ok := s.tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("embed: tensor %q not in safetensors file", name)
	}
	if t.Offsets[0] < 0 || t.Offsets[1] > int64(len(s.data)) || t.Offsets[0] > t.Offsets[1] {
		return nil, nil, fmt.Errorf("embed: tensor %q has offsets outside the data block", name)
	}
	raw := s.data[t.Offsets[0]:t.Offsets[1]]
	count := 1
	for _, d := range t.Shape {
		count *= d
	}
	out := make([]float32, count)
	switch t.DType {
	case "F32":
		if len(raw) != count*4 {
			return nil, nil, fmt.Errorf("embed: tensor %q: %d bytes for %d F32 values", name, len(raw), count)
		}
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case "F16":
		if len(raw) != count*2 {
			return nil, nil, fmt.Errorf("embed: tensor %q: %d bytes for %d F16 values", name, len(raw), count)
		}
		for i := range out {
			out[i] = float16To32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "BF16":
		if len(raw) != count*2 {
			return nil, nil, fmt.Errorf("embed: tensor %q: %d bytes for %d BF16 values", name, len(raw), count)
		}
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
	default:
		return nil, nil, fmt.Errorf("embed: tensor %q has unsupported dtype %s", name, t.DType)
	}
	return out, t.Shape, nil
}

// float16To32 expands an IEEE 754 half to a single.
func float16To32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: renormalise.
		shift := uint32(0)
		for frac&0x400 == 0 {
			frac <<= 1
			shift++
		}
		frac &= 0x3ff
		exp = 127 - 15 - shift + 1
		return math.Float32frombits(sign | exp<<23 | frac<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | frac<<13)
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | frac<<13)
	}
}

// marshalJSON is exposed for tests that need to build a safetensors header.
func marshalJSON(v any) ([]byte, error) { return json.Marshal(v) }
