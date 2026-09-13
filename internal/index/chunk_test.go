package index

import (
	"strings"
	"testing"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mirror"
)

func TestBuildChunksKeepsShortRowWhole(t *testing.T) {
	page := &mirror.PageRow{ID: "p1", Title: "Контейнер gateway держит 3,1 ГБ", Markdown: "\n"}
	chunks := BuildChunks(page, "Доверие: подтверждено\nТип: наблюдение")
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want one", len(chunks))
	}
	if !strings.Contains(chunks[0].Text, "Доверие: подтверждено") {
		t.Fatalf("property digest missing: %q", chunks[0].Text)
	}
	if !strings.Contains(chunks[0].Text, "gateway") {
		t.Fatalf("title missing: %q", chunks[0].Text)
	}
}

func TestBuildChunksSplitsOnHeadingsAndKeepsPath(t *testing.T) {
	md := `# Регламент

Вступление.

## §1. Модель данных

Пять слоёв.

### Процедурный

Эта страница.

## §2. Цикл

Читай, фильтруй.
`
	chunks := BuildChunks(&mirror.PageRow{ID: "p", Title: "Регламент", Markdown: md}, "")
	if len(chunks) < 3 {
		t.Fatalf("got %d chunks, expected one per section", len(chunks))
	}
	var found bool
	for _, c := range chunks {
		if strings.Contains(c.Heading, "§1. Модель данных › Процедурный") {
			found = true
		}
	}
	if !found {
		for _, c := range chunks {
			t.Logf("heading %q", c.Heading)
		}
		t.Fatal("nested heading path not built")
	}
}

func TestBuildChunksPutsTitleInFirstChunkOnly(t *testing.T) {
	md := "## A\n\nтекст A\n\n## B\n\nтекст B\n"
	chunks := BuildChunks(&mirror.PageRow{ID: "p", Title: "Заголовок", Markdown: md}, "")
	if !strings.Contains(chunks[0].Text, "Заголовок") {
		t.Fatalf("first chunk lacks the title: %q", chunks[0].Text)
	}
	for _, c := range chunks[1:] {
		if strings.Contains(c.Text, "Заголовок") {
			t.Fatalf("title repeated in chunk %d", c.Ord)
		}
	}
}

func TestSplitBySizeCutsOversizedSections(t *testing.T) {
	long := strings.Repeat("абзац с текстом. ", 200) // ~3400 runes
	pieces := splitBySize(long)
	if len(pieces) < 2 {
		t.Fatalf("got %d pieces for a %d-rune paragraph", len(pieces), len([]rune(long)))
	}
	for _, p := range pieces {
		if len([]rune(p)) > chunkMax {
			t.Fatalf("piece of %d runes exceeds the cap", len([]rune(p)))
		}
	}
}

func TestHeadingOfRecognisesOnlyRealHeadings(t *testing.T) {
	if l, _ := headingOf("## §1. Память"); l != 2 {
		t.Fatalf("level = %d", l)
	}
	if l, _ := headingOf("#hashtag"); l != 0 {
		t.Fatal("a hash without a space is not a heading")
	}
	if l, _ := headingOf("плain text"); l != 0 {
		t.Fatal("plain text is not a heading")
	}
}

func TestDigestValueMapsCheckboxSentinels(t *testing.T) {
	if got := digestValue("__YES__"); got != "да" {
		t.Fatalf("checked checkbox = %q", got)
	}
	if got := digestValue("__NO__"); got != "" {
		t.Fatalf("unchecked checkbox should add nothing, got %q", got)
	}
	if got := digestValue(12.0); got != "12" {
		t.Fatalf("number = %q", got)
	}
}

func TestLooksLikeIDList(t *testing.T) {
	if !looksLikeIDList(`["1f2e3d4c-5b6a-4798-8a9b-0c1d2e3f4a5b"]`) {
		t.Fatal("uuid list not detected")
	}
	if looksLikeIDList(`["idea","person"]`) {
		t.Fatal("tag list must not be treated as ids")
	}
}
