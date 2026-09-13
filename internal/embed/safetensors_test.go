package embed

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func writeSafetensors(t *testing.T, tensors map[string][]float32, shapes map[string][]int) string {
	t.Helper()
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
	header, err := marshalJSON(index)
	if err != nil {
		t.Fatal(err)
	}
	var out []byte
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(header)))
	out = append(out, lenBuf[:]...)
	out = append(out, header...)
	out = append(out, body...)

	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestSafetensorsRoundTrip(t *testing.T) {
	path := writeSafetensors(t,
		map[string][]float32{"a.weight": {1, 2, 3, 4}, "b.bias": {-1, 0.5}},
		map[string][]int{"a.weight": {2, 2}, "b.bias": {2}})

	st, err := OpenSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	got, shape, err := st.Float32("a.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 2 {
		t.Fatalf("shape = %v", shape)
	}
	for i, want := range []float32{1, 2, 3, 4} {
		if got[i] != want {
			t.Fatalf("value %d = %v, want %v", i, got[i], want)
		}
	}
	if _, _, err := st.Float32("missing"); err == nil {
		t.Fatal("expected an error for a missing tensor")
	}
}

func TestFloat16Conversion(t *testing.T) {
	cases := map[uint16]float32{
		0x0000: 0,
		0x3C00: 1,
		0xBC00: -1,
		0x4000: 2,
		0x3555: 0.333251953125,
	}
	for in, want := range cases {
		if got := float16To32(in); got != want {
			t.Errorf("float16To32(%#04x) = %v, want %v", in, got, want)
		}
	}
}
