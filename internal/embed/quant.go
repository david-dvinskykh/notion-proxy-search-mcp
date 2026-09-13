package embed

// QuantizeMatrix converts a float32 row-major [rows, cols] matrix to int8 with
// one symmetric scale per row. Per-row (per-output-channel) granularity is the
// cheapest scheme that keeps BERT projections within a couple of percent of
// float: a single scale for the whole tensor loses several percent because the
// output channels of attention projections differ in magnitude by an order.
func QuantizeMatrix(src []float32, rows, cols int) *QuantMatrix {
	out := &QuantMatrix{
		Rows:   rows,
		Cols:   cols,
		Data:   make([]int8, rows*cols),
		Scales: make([]float32, rows),
	}
	for r := 0; r < rows; r++ {
		row := src[r*cols : (r+1)*cols]
		maxAbs := float32(0)
		for _, v := range row {
			if v < 0 {
				v = -v
			}
			if v > maxAbs {
				maxAbs = v
			}
		}
		if maxAbs == 0 {
			continue
		}
		scale := maxAbs / 127
		out.Scales[r] = scale
		inv := 1 / scale
		dst := out.Data[r*cols : (r+1)*cols]
		for i, v := range row {
			q := nearestInt(v * inv)
			if q > 127 {
				q = 127
			} else if q < -127 {
				q = -127
			}
			dst[i] = int8(q)
		}
	}
	return out
}

// Dequantize expands a quantized matrix back to float32, used by tests and by
// the converter's self-check.
func Dequantize(m *QuantMatrix) []float32 {
	out := make([]float32, m.Rows*m.Cols)
	for r := 0; r < m.Rows; r++ {
		s := m.Scales[r]
		for c := 0; c < m.Cols; c++ {
			out[r*m.Cols+c] = float32(m.Data[r*m.Cols+c]) * s
		}
	}
	return out
}

// QuantizeVector packs a unit-length embedding into int8 with a single scale.
// Stored vectors are 4x smaller than float32 and the scan is a plain int32 dot
// product; for a few tens of thousands of chunks a brute-force scan beats any
// approximate index and has no build step to keep in sync.
func QuantizeVector(v []float32) (q []int8, scale float32) {
	maxAbs := float32(0)
	for _, x := range v {
		if x < 0 {
			x = -x
		}
		if x > maxAbs {
			maxAbs = x
		}
	}
	q = make([]int8, len(v))
	if maxAbs == 0 {
		return q, 0
	}
	scale = maxAbs / 127
	inv := 1 / scale
	for i, x := range v {
		qi := nearestInt(x * inv)
		if qi > 127 {
			qi = 127
		} else if qi < -127 {
			qi = -127
		}
		q[i] = int8(qi)
	}
	return q, scale
}

// DotQuantized returns the dot product of two int8-quantized vectors.
func DotQuantized(a []int8, aScale float32, b []int8, bScale float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	return float32(dotI8(a, b)) * aScale * bScale
}
