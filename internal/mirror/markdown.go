package mirror

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/notion"
)

// Node is a block together with the children fetched for it.
type Node struct {
	Block    notion.Block
	Children []Node
}

// RenderMarkdown turns a block tree into the Markdown the fetch tool returns.
// The shape follows Notion's own Markdown export closely enough that an agent
// reading it sees the same document it would see through the official server.
func RenderMarkdown(nodes []Node) string {
	var b strings.Builder
	renderNodes(&b, nodes, "", &listCounter{})
	return strings.TrimRight(b.String(), "\n") + "\n"
}

// listCounter keeps per-level numbering for ordered lists.
type listCounter struct{ n int }

func renderNodes(b *strings.Builder, nodes []Node, indent string, counter *listCounter) {
	for _, n := range nodes {
		if n.Block.Type != "numbered_list_item" {
			counter.n = 0
		}
		renderNode(b, n, indent, counter)
	}
}

func renderNode(b *strings.Builder, n Node, indent string, counter *listCounter) {
	t := n.Block.Type
	text := richTextOf(n.Block.Raw, t)

	switch t {
	case "paragraph":
		if strings.TrimSpace(text) != "" {
			writeBlock(b, indent, text)
		}
		renderChildren(b, n, indent+"\t")

	case "heading_1", "heading_2", "heading_3":
		level := int(t[len(t)-1] - '0')
		writeBlock(b, indent, strings.Repeat("#", level)+" "+text)
		renderChildren(b, n, indent)

	case "bulleted_list_item":
		writeLine(b, indent, "- "+text)
		renderChildren(b, n, indent+"\t")

	case "numbered_list_item":
		counter.n++
		writeLine(b, indent, strconv.Itoa(counter.n)+". "+text)
		renderChildren(b, n, indent+"\t")

	case "to_do":
		box := "[ ]"
		if boolField(n.Block.Raw, t, "checked") {
			box = "[x]"
		}
		writeLine(b, indent, "- "+box+" "+text)
		renderChildren(b, n, indent+"\t")

	case "toggle":
		writeBlock(b, indent, "<details>\n<summary>"+text+"</summary>")
		renderChildren(b, n, indent)
		writeBlock(b, indent, "</details>")

	case "quote":
		writeBlock(b, indent, "> "+strings.ReplaceAll(text, "\n", "\n> "))
		renderChildren(b, n, indent)

	case "callout":
		icon := calloutIcon(n.Block.Raw)
		writeBlock(b, indent, "> "+icon+strings.ReplaceAll(text, "\n", "\n> "))
		renderChildren(b, n, indent)

	case "code":
		lang := stringField(n.Block.Raw, t, "language")
		writeBlock(b, indent, "```"+lang+"\n"+text+"\n```")

	case "divider":
		writeBlock(b, indent, "---")

	case "equation":
		writeBlock(b, indent, "$$"+stringField(n.Block.Raw, t, "expression")+"$$")

	case "image", "video", "file", "pdf", "audio":
		url, caption := fileOf(n.Block.Raw, t)
		label := caption
		if label == "" {
			label = t
		}
		prefix := ""
		if t == "image" {
			prefix = "!"
		}
		writeBlock(b, indent, prefix+"["+label+"]("+url+")")

	case "bookmark", "embed", "link_preview":
		url := stringField(n.Block.Raw, t, "url")
		writeBlock(b, indent, "["+url+"]("+url+")")

	case "child_page":
		title := stringField(n.Block.Raw, t, "title")
		writeBlock(b, indent, fmt.Sprintf("<page id=%q>%s</page>", n.Block.ID, title))

	case "child_database":
		title := stringField(n.Block.Raw, t, "title")
		writeBlock(b, indent, fmt.Sprintf("<database id=%q>%s</database>", n.Block.ID, title))

	case "table":
		renderTable(b, indent, n)

	case "column_list", "column", "synced_block":
		renderChildren(b, n, indent)

	case "table_of_contents", "breadcrumb", "unsupported":
		// Nothing readable to mirror.

	default:
		if strings.TrimSpace(text) != "" {
			writeBlock(b, indent, text)
		}
		renderChildren(b, n, indent)
	}
}

func renderChildren(b *strings.Builder, n Node, indent string) {
	if len(n.Children) == 0 {
		return
	}
	renderNodes(b, n.Children, indent, &listCounter{})
}

// renderTable emits a GitHub-style table; the first row is the header when the
// block says it has one.
func renderTable(b *strings.Builder, indent string, n Node) {
	hasHeader := boolField(n.Block.Raw, "table", "has_column_header")
	rows := make([][]string, 0, len(n.Children))
	for _, child := range n.Children {
		if child.Block.Type != "table_row" {
			continue
		}
		rows = append(rows, tableRowCells(child.Block.Raw))
	}
	if len(rows) == 0 {
		return
	}
	width := 0
	for _, r := range rows {
		if len(r) > width {
			width = len(r)
		}
	}
	var out strings.Builder
	for i, r := range rows {
		for len(r) < width {
			r = append(r, "")
		}
		out.WriteString("| " + strings.Join(r, " | ") + " |\n")
		if i == 0 && hasHeader {
			out.WriteString("|" + strings.Repeat(" --- |", width) + "\n")
		}
	}
	writeBlock(b, indent, strings.TrimRight(out.String(), "\n"))
}

func tableRowCells(raw json.RawMessage) []string {
	var v struct {
		TableRow struct {
			Cells [][]notion.RichText `json:"cells"`
		} `json:"table_row"`
	}
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil
	}
	cells := make([]string, 0, len(v.TableRow.Cells))
	for _, c := range v.TableRow.Cells {
		cells = append(cells, strings.ReplaceAll(notion.Markdown(c), "|", `\|`))
	}
	return cells
}

// writeBlock writes a stand-alone block followed by a blank line.
func writeBlock(b *strings.Builder, indent, text string) {
	writeLine(b, indent, text)
	b.WriteString("\n")
}

// writeLine writes one indented line, indenting continuation lines too.
func writeLine(b *strings.Builder, indent, text string) {
	lines := strings.Split(text, "\n")
	for i, l := range lines {
		if i == 0 {
			b.WriteString(indent + l + "\n")
			continue
		}
		b.WriteString(indent + l + "\n")
	}
}

// richTextOf pulls the rich_text array out of whichever key the block uses.
func richTextOf(raw json.RawMessage, blockType string) string {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return ""
	}
	body, ok := envelope[blockType]
	if !ok {
		return ""
	}
	var fields struct {
		RichText []notion.RichText `json:"rich_text"`
		Caption  []notion.RichText `json:"caption"`
	}
	if err := json.Unmarshal(body, &fields); err != nil {
		return ""
	}
	if len(fields.RichText) > 0 {
		return notion.Markdown(fields.RichText)
	}
	return notion.Markdown(fields.Caption)
}

func stringField(raw json.RawMessage, blockType, field string) string {
	body := innerObject(raw, blockType)
	if body == nil {
		return ""
	}
	if s, ok := body[field].(string); ok {
		return s
	}
	return ""
}

func boolField(raw json.RawMessage, blockType, field string) bool {
	body := innerObject(raw, blockType)
	if body == nil {
		return false
	}
	b, _ := body[field].(bool)
	return b
}

func innerObject(raw json.RawMessage, blockType string) map[string]any {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil
	}
	body, ok := envelope[blockType]
	if !ok {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return nil
	}
	return m
}

func calloutIcon(raw json.RawMessage) string {
	body := innerObject(raw, "callout")
	if body == nil {
		return ""
	}
	icon, ok := body["icon"].(map[string]any)
	if !ok {
		return ""
	}
	if emoji, ok := icon["emoji"].(string); ok && emoji != "" {
		return emoji + " "
	}
	return ""
}

func fileOf(raw json.RawMessage, blockType string) (url, caption string) {
	body := innerObject(raw, blockType)
	if body == nil {
		return "", ""
	}
	if ext, ok := body["external"].(map[string]any); ok {
		url, _ = ext["url"].(string)
	}
	if url == "" {
		if f, ok := body["file"].(map[string]any); ok {
			url, _ = f["url"].(string)
		}
	}
	caption = richTextOf(raw, blockType)
	return url, caption
}
