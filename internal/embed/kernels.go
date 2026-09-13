package embed

import (
	"math"
	"runtime"
	"sync"
)

// quantizeRow converts one activation row to int8 with a symmetric per-row
// scale. Dynamic per-token quantization needs no calibration set, which
// matters here: there is no Russian/Ukrainian calibration corpus to hand.
func quantizeRow(dst []int8, src []float32) float32 {
	maxAbs := float32(0)
	for _, v := range src {
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return 0
	}
	scale := maxAbs / 127
	inv := 1 / scale
	for i, v := range src {
		q := nearestInt(v * inv)
		if q > 127 {
			q = 127
		} else if q < -127 {
			q = -127
		}
		dst[i] = int8(q)
	}
	return scale
}

// dotI8 is the inner kernel: an int8 dot product accumulated in int32.
//
// The shape of this loop is measured, not guessed. Hand-unrolling it into four
// or eight independent accumulators makes it roughly three times SLOWER than
// the plain ranged loop below, because the unrolled form defeats the
// compiler's own vectorisation and bounds-check elimination. Re-slicing b to
// len(a) is what removes the remaining bounds check. BenchmarkDotI8Length384
// is the arbiter: re-run it on the target board before changing this.
func dotI8(a, b []int8) int32 {
	b = b[:len(a)]
	var sum int32
	for i, x := range a {
		sum += int32(x) * int32(b[i])
	}
	return sum
}

// linear computes out = x * Wᵀ + bias for a [tokens, in] activation block.
// x is quantized per token, W carries one scale per output row, so the
// dequantisation is a single multiply per output element.
func linear(out []float32, x []float32, tokens int, w *QuantMatrix, bias []float32, scratch *scratchpad) {
	in := w.Cols
	rows := w.Rows
	qx := scratch.qact(tokens * in)
	scales := scratch.qscale(tokens)
	for t := 0; t < tokens; t++ {
		scales[t] = quantizeRow(qx[t*in:(t+1)*in], x[t*in:(t+1)*in])
	}

	// Tile the output so the stores stay contiguous. With the output index in
	// the inner position, out[t*rows+o] touches one cache line per token per
	// output; tiling the outputs and running the token loop outside the tile
	// turns that into a contiguous run of floats. Measured on a 384x1536
	// projection: 28.2 ms untiled, 15.3 ms at this tile width, single-threaded;
	// scaling across four workers is near linear either way. A tile of 128 was
	// the best of 16/32/64/128/256/384. BenchmarkLinear384x1536 is the arbiter
	// and should be re-run on the target board.
	const tile = 128
	parallelFor(rows, func(lo, hi int) {
		for o0 := lo; o0 < hi; o0 += tile {
			o1 := o0 + tile
			if o1 > hi {
				o1 = hi
			}
			for t := 0; t < tokens; t++ {
				qrow := qx[t*in : (t+1)*in : (t+1)*in]
				s := scales[t]
				row := out[t*rows+o0 : t*rows+o1]
				for o := o0; o < o1; o++ {
					acc := dotI8(qrow, w.Data[o*in:(o+1)*in:(o+1)*in])
					v := float32(acc) * w.Scales[o] * s
					if bias != nil {
						v += bias[o]
					}
					row[o-o0] = v
				}
			}
		}
	})
}

// layerNorm normalises each row in place.
func layerNorm(x []float32, tokens, dim int, gamma, beta []float32, eps float64) {
	for t := 0; t < tokens; t++ {
		row := x[t*dim : (t+1)*dim]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(dim)
		var variance float64
		for _, v := range row {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(dim)
		inv := 1 / math.Sqrt(variance+eps)
		for i, v := range row {
			row[i] = float32((float64(v)-mean)*inv)*gamma[i] + beta[i]
		}
	}
}

// gelu applies the exact Gaussian error linear unit used by BERT.
func gelu(x []float32) {
	for i, v := range x {
		x[i] = float32(0.5 * float64(v) * (1 + math.Erf(float64(v)/math.Sqrt2)))
	}
}

// softmaxRow turns logits into probabilities in place.
func softmaxRow(row []float32) {
	maxV := float32(math.Inf(-1))
	for _, v := range row {
		if v > maxV {
			maxV = v
		}
	}
	if math.IsInf(float64(maxV), -1) {
		return
	}
	var sum float32
	for i, v := range row {
		e := float32(math.Exp(float64(v - maxV)))
		row[i] = e
		sum += e
	}
	if sum == 0 {
		return
	}
	inv := 1 / sum
	for i := range row {
		row[i] *= inv
	}
}

// addInPlace computes a += b.
func addInPlace(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

// l2Normalize scales a vector to unit length; cosine similarity then reduces
// to a dot product, which is what the vector store relies on.
func l2Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

// workers caps parallelism. Four A72 cores also run other services, so leaving the
// default at GOMAXPROCS is deliberate but overridable.
var workers = runtime.GOMAXPROCS(0)

// SetWorkers limits how many goroutines the kernels use.
func SetWorkers(n int) {
	if n > 0 {
		workers = n
	}
}

// parallelFor splits [0,n) into contiguous bands, one per worker.
func parallelFor(n int, fn func(lo, hi int)) {
	w := workers
	if w > n {
		w = n
	}
	if w <= 1 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	band := (n + w - 1) / w
	for start := 0; start < n; start += band {
		end := start + band
		if end > n {
			end = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(start, end)
	}
	wg.Wait()
}

// scratchpad reuses activation buffers across forward passes so a long index
// run does not allocate per chunk.
type scratchpad struct {
	act    []int8
	scales []float32
	pool   map[int][]float32
}

func newScratchpad() *scratchpad { return &scratchpad{pool: map[int][]float32{}} }

func (s *scratchpad) qact(n int) []int8 {
	if cap(s.act) < n {
		s.act = make([]int8, n)
	}
	return s.act[:n]
}

func (s *scratchpad) qscale(n int) []float32 {
	if cap(s.scales) < n {
		s.scales = make([]float32, n)
	}
	return s.scales[:n]
}

// buf returns a named reusable float buffer of at least n elements.
func (s *scratchpad) buf(key, n int) []float32 {
	b := s.pool[key]
	if cap(b) < n {
		b = make([]float32, n)
		s.pool[key] = b
	}
	return b[:n]
}
