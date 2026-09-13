package embed

import (
	"encoding/json"
	"os"
	"testing"
)

// goldenCase is one reference segmentation produced by
// scripts/tokenizer_reference.py from the Hugging Face tokenizer.
type goldenCase struct {
	Text string  `json:"text"`
	IDs  []int32 `json:"ids"`
}

// testModel opens the weight blob named by NPS_TEST_MODEL, skipping the test
// when no blob is installed. The blob is ~118 MiB, so it is not committed; make
// model builds one.
func testModel(t *testing.T) *Model {
	t.Helper()
	path := os.Getenv("NPS_TEST_MODEL")
	if path == "" {
		t.Skip("set NPS_TEST_MODEL to a weight blob to run this test (make model)")
	}
	m, err := Load(path)
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// TestTokenizerMatchesReference is the fidelity contract for the tokenizer: the
// pure-Go Unigram implementation has to segment exactly as the tokenizer the
// checkpoint was trained with, across Russian, Ukrainian, Polish, English, SQL
// fragments, emoji and whitespace edge cases. It earned its keep: it caught
// trailing whitespace being trimmed, the dummy prefix being dropped on
// whitespace-only input, and zero-width separators being treated as text rather
// than as word boundaries.
func TestTokenizerMatchesReference(t *testing.T) {
	raw, err := os.ReadFile("testdata/tokenizer_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []goldenCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) < 50 {
		t.Fatalf("golden file holds only %d cases; regenerate it", len(cases))
	}

	m := testModel(t)
	var mismatches int
	for _, c := range cases {
		got := m.Tokenize(c.Text)
		if equalIDs(got, c.IDs) {
			continue
		}
		mismatches++
		if mismatches <= 5 {
			t.Errorf("text %q\n  reference: %v\n  got:       %v", c.Text, c.IDs, got)
		}
	}
	if mismatches > 0 {
		t.Fatalf("%d of %d cases diverge from the reference tokenizer", mismatches, len(cases))
	}
}

// TestControlCharactersAreDroppedDeliberately records the two measured
// divergences from the reference, both in a class that cannot reach the mirror
// through Notion's JSON API: the reference turns a NUL byte into <unk> and a
// word boundary, and keeps NEL (U+0085) as a piece of its own. Here both are
// dropped with the other control characters, which keeps a word whole.
func TestControlCharactersAreDroppedDeliberately(t *testing.T) {
	const wordWithPrefix = metaspace + "ab"
	for name, r := range map[string]rune{"NUL": 0x0000, "NEL": 0x0085, "ESC": 0x001B} {
		if got := Normalize("a" + string(r) + "b"); got != wordWithPrefix {
			t.Errorf("%s should be dropped without breaking the word, got %q", name, got)
		}
	}
	// Everything the reference treats as a boundary must be one here too.
	const twoWords = metaspace + "a" + metaspace + "b"
	for _, r := range []rune{0x200B, 0x200C, 0x200D, 0xFEFF, 0x00A0, 0x3000, 0x2028} {
		if got := Normalize("a" + string(r) + "b"); got != twoWords {
			t.Errorf("U+%04X should be a word boundary, got %q", r, got)
		}
	}
}

func equalIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
