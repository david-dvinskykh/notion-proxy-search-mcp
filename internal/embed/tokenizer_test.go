package embed

import (
	"testing"
)

// Two of these expectations were wrong until the reference tokenizer was
// consulted: trailing whitespace survives as a piece, and whitespace-only input
// still carries the dummy prefix. See TestTokenizerMatchesReference.
func TestNormalizeCollapsesSpaceAndAddsMetaspace(t *testing.T) {
	cases := map[string]string{
		"Привет мир":       "▁Привет▁мир",
		"  два   пробела ": "▁два▁пробела▁",
		"query: факт":      "▁query:▁факт",
		"\n\t":             "▁",
	}
	for in, want := range cases {
		if got := Normalize(in); got != want {
			t.Errorf("Normalize(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestNormalizeAppliesNFKC(t *testing.T) {
	// A composed and a decomposed "й" must normalise to the same bytes,
	// otherwise the same word indexes under two different segmentations.
	composed := Normalize("й")
	decomposed := Normalize("й")
	if composed != decomposed {
		t.Fatalf("NFKC not applied: %q vs %q", composed, decomposed)
	}
}

// buildTestVocab makes a tiny unigram vocabulary with the ids and scores laid
// out the way the converter writes them.
func buildTestVocab(t *testing.T, pieces map[string]float32) *Vocab {
	t.Helper()
	blob, err := buildVocabTensors(pieces, map[string]int32{})
	if err != nil {
		t.Fatal(err)
	}
	return blob
}

func TestViterbiPrefersHigherScoringSegmentation(t *testing.T) {
	v := buildTestVocab(t, map[string]float32{
		"▁":    -5,
		"▁к":   -6,
		"о":    -6,
		"т":    -6,
		"▁кот": -1,
	})
	ids := v.Encode("кот", 32)
	// BOS, the single "▁кот" piece, EOS.
	if len(ids) != 3 {
		t.Fatalf("got %d tokens %v, want the single-piece segmentation", len(ids), ids)
	}
}

func TestViterbiFallsBackToPieces(t *testing.T) {
	v := buildTestVocab(t, map[string]float32{
		"▁": -5, "к": -6, "о": -6, "т": -6,
	})
	ids := v.Encode("кот", 32)
	if len(ids) != 6 { // BOS + ▁ + к + о + т + EOS
		t.Fatalf("got %v", ids)
	}
}

func TestEncodeEmitsUnknownForUncoveredRunes(t *testing.T) {
	v := buildTestVocab(t, map[string]float32{"▁": -5})
	ids := v.Encode("猫", 32)
	found := false
	for _, id := range ids {
		if id == v.unkID {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected an unk token in %v", ids)
	}
}

func TestEncodeTruncatesToBudget(t *testing.T) {
	v := buildTestVocab(t, map[string]float32{"▁": -5, "а": -6})
	ids := v.Encode("а а а а а а а а", 6)
	if len(ids) != 6 {
		t.Fatalf("got %d tokens, want exactly the budget", len(ids))
	}
	if ids[0] != v.bos || ids[len(ids)-1] != v.eos {
		t.Fatalf("truncation must keep the special tokens: %v", ids)
	}
}

func TestLookupFindsEveryPiece(t *testing.T) {
	pieces := map[string]float32{"▁": -1, "▁дом": -2, "я": -3, "ё": -4}
	v := buildTestVocab(t, pieces)
	for p := range pieces {
		if _, _, ok := v.lookup([]byte(p)); !ok {
			t.Errorf("piece %q not found", p)
		}
	}
	if _, _, ok := v.lookup([]byte("отсутствует")); ok {
		t.Error("unexpected hit for a missing piece")
	}
}
