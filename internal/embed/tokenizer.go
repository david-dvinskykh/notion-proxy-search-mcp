package embed

import (
	"bytes"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// metaspace is the SentencePiece word boundary marker.
const metaspace = "▁"

// Vocab is a unigram SentencePiece vocabulary kept in the weight blob: pieces
// sorted lexicographically so a lookup is a binary search over mapped bytes,
// with no map to build at start-up. On a Raspberry Pi that difference is the
// whole start-up cost of the tokenizer.
type Vocab struct {
	blob    []byte
	offsets []uint32
	ids     []uint32
	scores  []float32

	maxPiece   int
	unkID      int32
	unkPenalty float64
	bos, eos   int32

	// added are the literal strings cut out of the input before normalization,
	// longest first so a match is leftmost-longest.
	added []addedToken
}

// addedToken is a string the reference tokenizer matches verbatim, ahead of
// normalization and segmentation: "<mask>" is one token there, never the six
// pieces its characters would otherwise make.
type addedToken struct {
	text string
	id   int32
}

// LoadVocab reads the vocabulary tensors out of a blob.
func LoadVocab(b *Blob) (*Vocab, error) {
	raw, err := b.Bytes("vocab.blob")
	if err != nil {
		return nil, err
	}
	offsets, _, err := b.U32("vocab.offsets")
	if err != nil {
		return nil, err
	}
	ids, _, err := b.U32("vocab.ids")
	if err != nil {
		return nil, err
	}
	scores, _, err := b.F32("vocab.scores")
	if err != nil {
		return nil, err
	}
	if len(offsets) != len(ids)+1 || len(ids) != len(scores) {
		return nil, fmt.Errorf("embed: vocabulary tensors disagree: %d offsets, %d ids, %d scores",
			len(offsets), len(ids), len(scores))
	}
	maxPiece := b.H.MaxPieceBytes
	if maxPiece <= 0 {
		maxPiece = 16
	}
	penalty := b.H.UnkPenalty
	if penalty == 0 {
		penalty = -10
	}
	added := make([]addedToken, 0, len(b.H.AddedTokens))
	for text, id := range b.H.AddedTokens {
		if text != "" {
			added = append(added, addedToken{text: text, id: int32(id)})
		}
	}
	sort.Slice(added, func(i, j int) bool {
		if len(added[i].text) != len(added[j].text) {
			return len(added[i].text) > len(added[j].text)
		}
		return added[i].text < added[j].text
	})
	return &Vocab{
		blob: raw, offsets: offsets, ids: ids, scores: scores,
		maxPiece: maxPiece, unkID: int32(b.H.UNK), unkPenalty: penalty,
		bos: int32(b.H.BOS), eos: int32(b.H.EOS), added: added,
	}, nil
}

// Size is the number of pieces.
func (v *Vocab) Size() int { return len(v.ids) }

// piece returns the bytes of the i-th sorted entry.
func (v *Vocab) piece(i int) []byte {
	return v.blob[v.offsets[i]:v.offsets[i+1]]
}

// lookup finds a piece by its exact bytes.
func (v *Vocab) lookup(p []byte) (id int32, score float32, ok bool) {
	i := sort.Search(len(v.ids), func(i int) bool {
		return bytes.Compare(v.piece(i), p) >= 0
	})
	if i >= len(v.ids) || !bytes.Equal(v.piece(i), p) {
		return 0, 0, false
	}
	return int32(v.ids[i]), v.scores[i], true
}

// Normalize applies the preprocessing the model was trained with: NFKC,
// collapsed whitespace, then the metaspace word marker.
func Normalize(text string) string {
	text = norm.NFKC.String(text)
	if text == "" {
		return ""
	}
	var b strings.Builder
	b.Grow(len(text) + 3)
	space := true // leading whitespace is dropped, as in SentencePiece
	for _, r := range text {
		switch {
		case spaceLike(r):
			if !space {
				b.WriteByte(' ')
				space = true
			}
		case unicode.Is(unicode.Cc, r):
			// Control characters are removed and do not break a word: the
			// reference tokenizer answers a single piece for a word with an
			// escape or bell byte in the middle of it.
		default:
			b.WriteRune(r)
			space = false
		}
	}
	// Two rules taken from the reference tokenizer rather than guessed:
	// trailing whitespace is collapsed to one space and KEPT (it becomes a
	// standalone metaspace piece), and a non-empty input always carries the
	// dummy prefix even when every character in it was whitespace: a lone space
	// and a lone newline both tokenize to one metaspace piece. Only a genuinely
	// empty string yields no pieces at all.
	return metaspace + strings.ReplaceAll(b.String(), " ", metaspace)
}

// spaceLike reports which characters become a word boundary. The set was read
// off the reference tokenizer, not inferred: it is Unicode whitespace plus the
// zero-width separators that turn up in pasted text (ZWSP, ZWNJ, ZWJ and the
// byte-order mark, all of which the reference maps to a boundary), minus NEL,
// which the reference keeps as a piece of its own.
//
// Two measured divergences remain, both in a class that cannot reach the mirror
// through Notion's API: a NUL byte becomes <unk> and a boundary in the
// reference, and NEL (U+0085) keeps its own piece there; here both are dropped
// with the other control characters. TestTokenizerMatchesReference records the
// agreement on everything else.
func spaceLike(r rune) bool {
	switch r {
	case 0x200B, 0x200C, 0x200D, 0xFEFF:
		return true
	case 0x0085:
		return false
	}
	return unicode.IsSpace(r)
}

// Encode turns text into token ids, wrapped in the model's BOS/EOS tokens and
// truncated to maxTokens including them.
func (v *Vocab) Encode(text string, maxTokens int) []int32 {
	pieces := v.segment(text)
	budget := maxTokens - 2
	if budget < 0 {
		budget = 0
	}
	if len(pieces) > budget {
		pieces = pieces[:budget]
	}
	out := make([]int32, 0, len(pieces)+2)
	out = append(out, v.bos)
	out = append(out, pieces...)
	return append(out, v.eos)
}

// segment splits the text on the added tokens and runs the unigram decoder on
// what lies between them. The reference tokenizer does the cut first and then
// normalizes each part on its own, so every part carries its own dummy prefix:
// "a<mask>b" is ["▁a", "<mask>", "▁b"], not ["▁a", "<", "mask", ">", "b"].
func (v *Vocab) segment(text string) []int32 {
	if len(v.added) == 0 {
		return v.viterbi(Normalize(text))
	}
	var out []int32
	for text != "" {
		at, tok := v.nextAdded(text)
		if at < 0 {
			return append(out, v.viterbi(Normalize(text))...)
		}
		out = append(out, v.viterbi(Normalize(text[:at]))...)
		out = append(out, tok.id)
		text = text[at+len(tok.text):]
	}
	return out
}

// nextAdded finds the leftmost added token in text, preferring the longest one
// that starts there. It returns -1 when the text holds none.
func (v *Vocab) nextAdded(text string) (int, addedToken) {
	best := -1
	var found addedToken
	for _, a := range v.added {
		i := strings.Index(text, a.text)
		if i < 0 || (best >= 0 && i >= best) {
			// v.added is longest first, so an equal position keeps the longer
			// match already recorded.
			continue
		}
		best, found = i, a
	}
	return best, found
}

// viterbi finds the highest-scoring segmentation of s, the standard unigram
// SentencePiece decoding.
func (v *Vocab) viterbi(s string) []int32 {
	n := len(s)
	if n == 0 {
		return nil
	}
	const negInf = -math.MaxFloat64
	best := make([]float64, n+1)
	prev := make([]int, n+1)
	token := make([]int32, n+1)
	for i := 1; i <= n; i++ {
		best[i] = negInf
		prev[i] = -1
	}

	for i := 0; i < n; i++ {
		if best[i] == negInf {
			continue
		}
		if !utf8.RuneStart(s[i]) {
			continue
		}
		limit := i + v.maxPiece
		if limit > n {
			limit = n
		}
		matched := false
		for j := i + 1; j <= limit; j++ {
			if j < n && !utf8.RuneStart(s[j]) {
				continue
			}
			id, score, ok := v.lookup([]byte(s[i:j]))
			if !ok {
				continue
			}
			matched = true
			if cand := best[i] + float64(score); cand > best[j] {
				best[j] = cand
				prev[j] = i
				token[j] = id
			}
		}
		if !matched {
			// No piece covers even one rune here: spend an <unk> on it so the
			// rest of the string still gets segmented.
			_, size := utf8.DecodeRuneInString(s[i:])
			j := i + size
			if cand := best[i] + v.unkPenalty; cand > best[j] {
				best[j] = cand
				prev[j] = i
				token[j] = v.unkID
			}
		}
	}

	if prev[n] < 0 {
		return []int32{v.unkID}
	}
	var rev []int32
	for i := n; i > 0; i = prev[i] {
		rev = append(rev, token[i])
		if prev[i] < 0 {
			break
		}
	}
	for l, r := 0, len(rev)-1; l < r; l, r = l+1, r-1 {
		rev[l], rev[r] = rev[r], rev[l]
	}
	return rev
}
