package notion

import "testing"

func spanWith(text string, mutate func(*RichText)) RichText {
	s := RichText{Type: "text", PlainText: text}
	if mutate != nil {
		mutate(&s)
	}
	return s
}

func TestMarkdownKeepsEmphasisNextToWords(t *testing.T) {
	spans := []RichText{
		spanWith("plain ", nil),
		spanWith("bold ", func(s *RichText) { s.Annotations.Bold = true }),
		spanWith("tail", nil),
	}
	got := Markdown(spans)
	want := "plain **bold** tail"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestMarkdownCodeSpanIsNotNested(t *testing.T) {
	spans := []RichText{spanWith("x := 1", func(s *RichText) {
		s.Annotations.Code = true
		s.Annotations.Bold = true
	})}
	if got := Markdown(spans); got != "`x := 1`" {
		t.Fatalf("got %q", got)
	}
}

func TestMarkdownRendersLinksAndEquations(t *testing.T) {
	spans := []RichText{
		spanWith("docs", func(s *RichText) { s.Href = "https://example.com" }),
		{Type: "equation", Equation: struct {
			Expression string `json:"expression"`
		}{Expression: "a^2"}},
	}
	if got := Markdown(spans); got != "[docs](https://example.com)$a^2$" {
		t.Fatalf("got %q", got)
	}
}

func TestCleanExtractsIDFromURL(t *testing.T) {
	cases := map[string]string{
		"https://app.notion.com/p/1f2e3d4c5b6a47988a9b0c1d2e3f4a5b?pvs=204":           "1f2e3d4c5b6a47988a9b0c1d2e3f4a5b",
		"https://www.notion.so/workspace/Page-Title-9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e": "9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e",
		"1f2e3d4c-5b6a-4798-8a9b-0c1d2e3f4a5b":                                        "1f2e3d4c-5b6a-4798-8a9b-0c1d2e3f4a5b",
	}
	for in, want := range cases {
		if got := clean(in); got != want {
			t.Errorf("clean(%q) = %q, want %q", in, got, want)
		}
	}
}
