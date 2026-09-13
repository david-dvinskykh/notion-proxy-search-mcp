// Package index turns mirrored pages into retrieval units and answers queries
// over them: BM25 from SQLite's FTS5, cosine over int8 vectors, reciprocal rank
// fusion of the two, and one hop of relation-graph expansion.
package index

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mirror"
)

// Chunk is one retrieval unit.
type Chunk struct {
	ID      int64
	PageID  string
	Ord     int
	Heading string
	Text    string
}

// chunkTarget is the size a page is split towards, in runes. Database rows are
// usually one sentence and stay whole; long pages split on headings first and
// only then on size, so a chunk keeps a complete thought.
//
// 500 runes is measured, not picked: the real tokenizer turns Russian prose into
// one token per 2.8-3.2 runes, so a chunk of this size is about 185 tokens. Embedding cost is linear in total tokens, so chunk width
// barely moves the backfill total — but it sets per-chunk latency (185 tokens
// costs ~1 s on four x86 cores, ~300 tokens costs ~1.7 s) and retrieval
// granularity, and both argue for the smaller unit.
const chunkTarget = 500

// chunkMax caps a chunk so one enormous paragraph cannot blow past the model's
// token window.
const chunkMax = 800

// BuildChunks splits a mirrored page into retrieval units. The first chunk
// always carries the page title and its property digest, so a database row is
// findable by its own fields even when the body is empty.
func BuildChunks(page *mirror.PageRow, propDigest string) []Chunk {
	header := strings.TrimSpace(page.Title)
	if propDigest != "" {
		if header != "" {
			header += "\n"
		}
		header += propDigest
	}

	sections := splitSections(page.Markdown)
	var out []Chunk
	add := func(heading, text string) {
		text = strings.TrimSpace(text)
		if text == "" {
			return
		}
		out = append(out, Chunk{PageID: page.ID, Ord: len(out), Heading: heading, Text: text})
	}

	if len(sections) == 0 {
		add("", header)
		return out
	}
	for i, s := range sections {
		body := s.body
		if i == 0 && header != "" {
			body = header + "\n\n" + body
		}
		for _, piece := range splitBySize(body) {
			add(s.heading, piece)
		}
	}
	if len(out) == 0 {
		add("", header)
	}
	return out
}

type section struct {
	heading string
	body    string
}

// splitSections cuts Markdown at headings, keeping the heading path as context.
func splitSections(md string) []section {
	lines := strings.Split(md, "\n")
	var out []section
	var path []string
	var body strings.Builder

	flush := func() {
		if strings.TrimSpace(body.String()) == "" {
			body.Reset()
			return
		}
		out = append(out, section{heading: strings.Join(path, " › "), body: body.String()})
		body.Reset()
	}

	for _, line := range lines {
		level, title := headingOf(line)
		if level == 0 {
			body.WriteString(line)
			body.WriteString("\n")
			continue
		}
		flush()
		if level > len(path) {
			path = append(path, title)
		} else {
			path = append(path[:level-1], title)
		}
	}
	flush()
	return out
}

// headingOf returns the level and text of a Markdown ATX heading.
func headingOf(line string) (int, string) {
	trimmed := strings.TrimLeft(line, " \t")
	level := 0
	for level < len(trimmed) && trimmed[level] == '#' {
		level++
	}
	if level == 0 || level > 6 || level >= len(trimmed) || trimmed[level] != ' ' {
		return 0, ""
	}
	return level, strings.TrimSpace(trimmed[level+1:])
}

// splitBySize breaks an oversized section on paragraph boundaries.
func splitBySize(body string) []string {
	if len([]rune(body)) <= chunkMax {
		return []string{body}
	}
	paragraphs := strings.Split(body, "\n\n")
	var out []string
	var cur strings.Builder
	for _, p := range paragraphs {
		if cur.Len() > 0 && len([]rune(cur.String()))+len([]rune(p)) > chunkTarget {
			out = append(out, cur.String())
			cur.Reset()
		}
		if len([]rune(p)) > chunkMax {
			// A single paragraph longer than the cap: cut it on rune bounds.
			runes := []rune(p)
			for start := 0; start < len(runes); start += chunkTarget {
				end := start + chunkTarget
				if end > len(runes) {
					end = len(runes)
				}
				out = append(out, string(runes[start:end]))
			}
			continue
		}
		if cur.Len() > 0 {
			cur.WriteString("\n\n")
		}
		cur.WriteString(p)
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}

// PropertyDigest renders a row's properties as "Name: value" lines, skipping
// the machine-only ones. It is what makes a database row searchable by the
// fields an agent actually filters on.
//
// The page title is skipped wherever it appears as a property value: in a
// database of one-sentence facts the title IS the statement, and emitting it
// again under its property name tripled the text of every row — the same
// sentence as the chunk header, as a digest line, and sometimes in the body.
// Embedding cost is linear in tokens, so that was a third of the backfill.
func PropertyDigest(ctx context.Context, db *sql.DB, pageID, title string) (string, error) {
	rows, err := db.QueryContext(ctx,
		`SELECT col, value FROM page_props WHERE page_id=? AND value IS NOT NULL ORDER BY col`, pageID)
	if err != nil {
		return "", err
	}
	defer rows.Close()
	var b strings.Builder
	for rows.Next() {
		var col string
		var value any
		if err := rows.Scan(&col, &value); err != nil {
			return "", err
		}
		text := digestValue(value)
		if text == "" || text == title {
			continue
		}
		// Relation and people columns hold opaque ids; they are edges, not text.
		if strings.HasPrefix(text, `["`) && strings.Contains(text, "-") && looksLikeIDList(text) {
			continue
		}
		if strings.HasPrefix(col, "date:") && strings.HasSuffix(col, ":is_datetime") {
			continue
		}
		b.WriteString(displayColumn(col))
		b.WriteString(": ")
		b.WriteString(text)
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n"), rows.Err()
}

func displayColumn(col string) string {
	if rest, ok := strings.CutPrefix(col, "date:"); ok {
		if name, ok := strings.CutSuffix(rest, ":start"); ok {
			return name
		}
		if name, ok := strings.CutSuffix(rest, ":end"); ok {
			return name + " (до)"
		}
	}
	if rest, ok := strings.CutPrefix(col, "userDefined:"); ok {
		return rest
	}
	return col
}

func digestValue(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		switch t {
		case "__YES__":
			return "да"
		case "__NO__":
			return ""
		case "[]", "{}":
			return ""
		}
		return t
	case int64:
		return fmt.Sprint(t)
	case float64:
		return strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.4f", t), "0"), ".")
	default:
		return fmt.Sprint(t)
	}
}

// looksLikeIDList reports whether a JSON array holds only UUID-shaped strings.
func looksLikeIDList(s string) bool {
	body := strings.Trim(s, "[]")
	if body == "" {
		return false
	}
	for _, part := range strings.Split(body, ",") {
		p := strings.Trim(strings.TrimSpace(part), `"`)
		if len(p) != 36 && len(p) != 32 {
			return false
		}
	}
	return true
}
