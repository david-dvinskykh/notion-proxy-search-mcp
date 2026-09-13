package notion

import (
	"encoding/json"
	"strings"
)

// RichText is one span of Notion rich text.
type RichText struct {
	Type        string `json:"type"`
	PlainText   string `json:"plain_text"`
	Href        string `json:"href"`
	Annotations struct {
		Bold          bool   `json:"bold"`
		Italic        bool   `json:"italic"`
		Strikethrough bool   `json:"strikethrough"`
		Underline     bool   `json:"underline"`
		Code          bool   `json:"code"`
		Color         string `json:"color"`
	} `json:"annotations"`
	Mention  json.RawMessage `json:"mention"`
	Equation struct {
		Expression string `json:"expression"`
	} `json:"equation"`
}

// PlainTextOf joins the plain text of a span list.
func PlainTextOf(spans []RichText) string {
	var b strings.Builder
	for _, s := range spans {
		b.WriteString(s.PlainText)
	}
	return b.String()
}

// Markdown renders spans with inline formatting preserved.
func Markdown(spans []RichText) string {
	var b strings.Builder
	for _, s := range spans {
		text := s.PlainText
		if s.Type == "equation" && s.Equation.Expression != "" {
			b.WriteString("$" + s.Equation.Expression + "$")
			continue
		}
		if text == "" {
			continue
		}
		// Code wins over the other annotations: Notion does not nest emphasis
		// inside code spans and neither does Markdown.
		if s.Annotations.Code {
			b.WriteString("`" + text + "`")
			continue
		}
		leading, core, trailing := splitSpace(text)
		if core == "" {
			b.WriteString(text)
			continue
		}
		wrapped := core
		if s.Annotations.Bold {
			wrapped = "**" + wrapped + "**"
		}
		if s.Annotations.Italic {
			wrapped = "_" + wrapped + "_"
		}
		if s.Annotations.Strikethrough {
			wrapped = "~~" + wrapped + "~~"
		}
		if s.Href != "" {
			wrapped = "[" + wrapped + "](" + s.Href + ")"
		}
		b.WriteString(leading + wrapped + trailing)
	}
	return b.String()
}

// splitSpace peels surrounding whitespace so emphasis markers sit next to the
// words they apply to; "** bold** " does not render as bold in Markdown.
func splitSpace(s string) (leading, core, trailing string) {
	i := 0
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	j := len(s)
	for j > i && (s[j-1] == ' ' || s[j-1] == '\t') {
		j--
	}
	return s[:i], s[i:j], s[j:]
}
