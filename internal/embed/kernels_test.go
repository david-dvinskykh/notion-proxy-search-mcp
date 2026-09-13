package embed

import (
	"math"
	"math/rand"
	"testing"
)

func TestDotI8MatchesNaiveSum(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 7, 8, 9, 64, 383, 384, 1536} {
		a := make([]int8, n)
		b := make([]int8, n)
		var want int32
		for i := 0; i < n; i++ {
			a[i] = int8(rng.Intn(255) - 127)
			b[i] = int8(rng.Intn(255) - 127)
			want += int32(a[i]) * int32(b[i])
		}
		if got := dotI8(a, b); got != want {
			t.Fatalf("n=%d: dotI8 = %d, want %d", n, got, want)
		}
	}
}

// TestLinearApproximatesFloatMatmul is the accuracy budget for int8: a
// quantized projection must stay within a fraction of a percent of the float
// result, otherwise ranking quality degrades rather than just shifting.
func TestLinearApproximatesFloatMatmul(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const tokens, in, out = 5, 384, 256

	wf := make([]float32, out*in)
	for i := range wf {
		wf[i] = float32(rng.NormFloat64() * 0.05)
	}
	x := make([]float32, tokens*in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	bias := make([]float32, out)
	for i := range bias {
		bias[i] = float32(rng.NormFloat64() * 0.01)
	}

	qm := QuantizeMatrix(wf, out, in)
	got := make([]float32, tokens*out)
	linear(got, x, tokens, qm, bias, newScratchpad())

	var num, den float64
	for t0 := 0; t0 < tokens; t0++ {
		for o := 0; o < out; o++ {
			var ref float64
			for k := 0; k < in; k++ {
				ref += float64(x[t0*in+k]) * float64(wf[o*in+k])
			}
			ref += float64(bias[o])
			d := ref - float64(got[t0*out+o])
			num += d * d
			den += ref * ref
		}
	}
	rel := math.Sqrt(num / den)
	if rel > 0.02 {
		t.Fatalf("relative error %.4f exceeds the 2%% int8 budget", rel)
	}
}

func TestQuantizeRowRoundTrip(t *testing.T) {
	src := []float32{-3, -0.5, 0, 0.25, 3}
	dst := make([]int8, len(src))
	scale := quantizeRow(dst, src)
	for i, v := range src {
		got := float32(dst[i]) * scale
		if math.Abs(float64(got-v)) > float64(scale) {
			t.Fatalf("index %d: %v -> %v (scale %v)", i, v, got, scale)
		}
	}
}

func TestQuantizeRowAllZeros(t *testing.T) {
	dst := make([]int8, 4)
	if scale := quantizeRow(dst, []float32{0, 0, 0, 0}); scale != 0 {
		t.Fatalf("scale = %v, want 0", scale)
	}
}

func TestLayerNormZeroMeanUnitVariance(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	gamma := []float32{1, 1, 1, 1}
	beta := []float32{0, 0, 0, 0}
	layerNorm(x, 1, 4, gamma, beta, 1e-12)
	var mean, variance float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= 4
	for _, v := range x {
		variance += (float64(v) - mean) * (float64(v) - mean)
	}
	variance /= 4
	if math.Abs(mean) > 1e-5 || math.Abs(variance-1) > 1e-4 {
		t.Fatalf("mean %v variance %v", mean, variance)
	}
}

func TestSoftmaxRowSumsToOne(t *testing.T) {
	row := []float32{1, 2, 3}
	softmaxRow(row)
	var sum float32
	for _, v := range row {
		sum += v
	}
	if math.Abs(float64(sum)-1) > 1e-6 {
		t.Fatalf("sum = %v", sum)
	}
	if row[2] <= row[1] || row[1] <= row[0] {
		t.Fatal("softmax must preserve order")
	}
}

func TestGeluMatchesReference(t *testing.T) {
	x := []float32{-2, -1, 0, 1, 2}
	want := []float32{-0.04550, -0.15866, 0, 0.84134, 1.95450}
	gelu(x)
	for i := range x {
		if math.Abs(float64(x[i]-want[i])) > 1e-4 {
			t.Fatalf("gelu[%d] = %v, want %v", i, x[i], want[i])
		}
	}
}

func TestL2NormalizeMakesUnitVector(t *testing.T) {
	v := []float32{3, 4}
	l2Normalize(v)
	if math.Abs(float64(v[0]*v[0]+v[1]*v[1])-1) > 1e-6 {
		t.Fatalf("not unit length: %v", v)
	}
}

func BenchmarkDotI8Length384(b *testing.B) {
	x := make([]int8, 384)
	y := make([]int8, 384)
	for i := range x {
		x[i], y[i] = int8(i%127), int8((i*7)%127)
	}
	b.SetBytes(384)
	for i := 0; i < b.N; i++ {
		_ = dotI8(x, y)
	}
}

func BenchmarkLinear384x1536(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	wf := make([]float32, 1536*384)
	for i := range wf {
		wf[i] = float32(rng.NormFloat64() * 0.05)
	}
	qm := QuantizeMatrix(wf, 1536, 384)
	const tokens = 32
	x := make([]float32, tokens*384)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	out := make([]float32, tokens*1536)
	sp := newScratchpad()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		linear(out, x, tokens, qm, nil, sp)
	}
}
