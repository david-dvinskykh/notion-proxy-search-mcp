package index

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
)

// EvalCase is one question with the page that should answer it. Expect holds a
// page id or any Notion URL containing it, so cases can be pasted straight out
// of a browser.
type EvalCase struct {
	Question string `json:"q"`
	Expect   string `json:"expect"`
	Note     string `json:"note,omitempty"`
}

// LoadEvalCases reads a JSON array of cases.
func LoadEvalCases(path string) ([]EvalCase, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cases []EvalCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	for i, c := range cases {
		if strings.TrimSpace(c.Question) == "" || strings.TrimSpace(c.Expect) == "" {
			return nil, fmt.Errorf("case %d needs both q and expect", i+1)
		}
	}
	return cases, nil
}

// pageIDPattern matches a Notion id with or without dashes. Anchoring on the
// 8-4-4-4-12 shape rather than "every hex character" matters: a URL slug is full
// of hex-looking letters and the query string ends in digits, and concatenating
// those produced a plausible-looking id that matched nothing.
var pageIDPattern = regexp.MustCompile(`(?i)[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}`)

// CanonicalPageID reduces a page id or Notion URL to 32 lowercase hex digits, so
// dashed ids, bare ids and URLs with a slug all compare equal. The last match
// wins: Notion puts the id at the end of the URL, after the slug.
func CanonicalPageID(s string) string {
	matches := pageIDPattern.FindAllString(s, -1)
	if len(matches) == 0 {
		return ""
	}
	last := matches[len(matches)-1]
	return strings.ToLower(strings.ReplaceAll(last, "-", ""))
}

// RankOf returns the 1-based position of the expected page in results, or 0.
func RankOf(results []Result, expect string) int {
	want := CanonicalPageID(expect)
	for i, r := range results {
		if CanonicalPageID(r.PageID) == want {
			return i + 1
		}
	}
	return 0
}

// EvalMetrics summarises one mode over a case set.
type EvalMetrics struct {
	Mode    string  `json:"mode"`
	Cases   int     `json:"cases"`
	Hit1    int     `json:"hit_at_1"`
	Hit3    int     `json:"hit_at_3"`
	Hit5    int     `json:"hit_at_5"`
	Missed  int     `json:"missed"`
	MRR     float64 `json:"mrr"`
	Ranks   []int   `json:"ranks"`
	Slowest string  `json:"slowest_case,omitempty"`
}

// Summarise turns per-case ranks (0 = not found) into metrics.
func Summarise(mode string, ranks []int) EvalMetrics {
	m := EvalMetrics{Mode: mode, Cases: len(ranks), Ranks: ranks}
	for _, r := range ranks {
		switch {
		case r == 0:
			m.Missed++
			continue
		case r == 1:
			m.Hit1++
		}
		if r <= 3 {
			m.Hit3++
		}
		if r <= 5 {
			m.Hit5++
		}
		m.MRR += 1 / float64(r)
	}
	if len(ranks) > 0 {
		m.MRR /= float64(len(ranks))
	}
	return m
}

// FormatEval renders a compact report: one line per mode, then the cases that
// no mode answered at rank one, which are the ones worth reading.
func FormatEval(all []EvalMetrics, cases []EvalCase) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%-8s %6s %6s %6s %6s %7s\n", "mode", "hit@1", "hit@3", "hit@5", "miss", "MRR")
	for _, m := range all {
		fmt.Fprintf(&b, "%-8s %5d%% %5d%% %5d%% %6d %7.3f\n", m.Mode,
			pct(m.Hit1, m.Cases), pct(m.Hit3, m.Cases), pct(m.Hit5, m.Cases), m.Missed, m.MRR)
	}
	hard := []string{}
	for i, c := range cases {
		best := 0
		for _, m := range all {
			if i < len(m.Ranks) && m.Ranks[i] > 0 && (best == 0 || m.Ranks[i] < best) {
				best = m.Ranks[i]
			}
		}
		if best != 1 {
			where := "not found"
			if best > 0 {
				where = fmt.Sprintf("best rank %d", best)
			}
			hard = append(hard, fmt.Sprintf("  %-9s %s", where, c.Question))
		}
	}
	if len(hard) > 0 {
		sort.Strings(hard)
		fmt.Fprintf(&b, "\nnot answered first by any mode (%d of %d):\n%s\n",
			len(hard), len(cases), strings.Join(hard, "\n"))
	}
	return b.String()
}

func pct(n, total int) int {
	if total == 0 {
		return 0
	}
	return int(float64(n)/float64(total)*100 + 0.5)
}
