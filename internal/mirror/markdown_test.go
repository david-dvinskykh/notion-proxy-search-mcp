package mirror

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/notion"
)

func block(t *testing.T, body string) notion.Block {
	t.Helper()
	var b notion.Block
	if err := json.Unmarshal([]byte(body), &b); err != nil {
		t.Fatalf("parse block: %v", err)
	}
	return b
}

func text(s string) string {
	return `[{"type":"text","plain_text":"` + s + `","annotations":{}}]`
}

func TestRenderHeadingsParagraphsAndLists(t *testing.T) {
	nodes := []Node{
		{Block: block(t, `{"id":"1","type":"heading_2","heading_2":{"rich_text":`+text("Раздел")+`}}`)},
		{Block: block(t, `{"id":"2","type":"paragraph","paragraph":{"rich_text":`+text("Текст")+`}}`)},
		{Block: block(t, `{"id":"3","type":"bulleted_list_item","bulleted_list_item":{"rich_text":`+text("первый")+`}}`)},
		{Block: block(t, `{"id":"4","type":"numbered_list_item","numbered_list_item":{"rich_text":`+text("шаг один")+`}}`)},
		{Block: block(t, `{"id":"5","type":"numbered_list_item","numbered_list_item":{"rich_text":`+text("шаг два")+`}}`)},
	}
	got := RenderMarkdown(nodes)
	for _, want := range []string{"## Раздел", "- первый", "1. шаг один", "2. шаг два"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
}

func TestRenderNestedChildrenAreIndented(t *testing.T) {
	nodes := []Node{{
		Block: block(t, `{"id":"1","type":"bulleted_list_item","bulleted_list_item":{"rich_text":`+text("верх")+`}}`),
		Children: []Node{{
			Block: block(t, `{"id":"2","type":"bulleted_list_item","bulleted_list_item":{"rich_text":`+text("низ")+`}}`),
		}},
	}}
	got := RenderMarkdown(nodes)
	if !strings.Contains(got, "\t- низ") {
		t.Fatalf("child not indented:\n%q", got)
	}
}

func TestRenderTableWithHeader(t *testing.T) {
	nodes := []Node{{
		Block: block(t, `{"id":"t","type":"table","table":{"table_width":2,"has_column_header":true}}`),
		Children: []Node{
			{Block: block(t, `{"id":"r1","type":"table_row","table_row":{"cells":[`+text("Ось")+`,`+text("Вердикт")+`]}}`)},
			{Block: block(t, `{"id":"r2","type":"table_row","table_row":{"cells":[`+text("Извлечение")+`,`+text("отстаёт")+`]}}`)},
		},
	}}
	got := RenderMarkdown(nodes)
	if !strings.Contains(got, "| Ось | Вердикт |") || !strings.Contains(got, "| --- | --- |") {
		t.Fatalf("table not rendered:\n%s", got)
	}
}

func TestRenderCodeKeepsLanguage(t *testing.T) {
	nodes := []Node{{Block: block(t, `{"id":"c","type":"code","code":{"language":"go","rich_text":`+text("package main")+`}}`)}}
	got := RenderMarkdown(nodes)
	if !strings.Contains(got, "```go\npackage main\n```") {
		t.Fatalf("code block:\n%s", got)
	}
}

func TestRenderCalloutKeepsEmoji(t *testing.T) {
	nodes := []Node{{Block: block(t, `{"id":"c","type":"callout","callout":{"icon":{"type":"emoji","emoji":"🔒"},"rich_text":`+text("защищено")+`}}`)}}
	if got := RenderMarkdown(nodes); !strings.Contains(got, "> 🔒 защищено") {
		t.Fatalf("callout:\n%s", got)
	}
}

func TestRenderToDoCheckbox(t *testing.T) {
	nodes := []Node{
		{Block: block(t, `{"id":"a","type":"to_do","to_do":{"checked":true,"rich_text":`+text("сделано")+`}}`)},
		{Block: block(t, `{"id":"b","type":"to_do","to_do":{"checked":false,"rich_text":`+text("нет")+`}}`)},
	}
	got := RenderMarkdown(nodes)
	if !strings.Contains(got, "- [x] сделано") || !strings.Contains(got, "- [ ] нет") {
		t.Fatalf("to_do:\n%s", got)
	}
}

func TestRenderColumnsAreFlattened(t *testing.T) {
	nodes := []Node{{
		Block: block(t, `{"id":"cl","type":"column_list","column_list":{}}`),
		Children: []Node{{
			Block:    block(t, `{"id":"c1","type":"column","column":{}}`),
			Children: []Node{{Block: block(t, `{"id":"p","type":"paragraph","paragraph":{"rich_text":`+text("внутри колонки")+`}}`)}},
		}},
	}}
	if got := RenderMarkdown(nodes); !strings.Contains(got, "внутри колонки") {
		t.Fatalf("column content lost:\n%s", got)
	}
}
