package mirror

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/notion"
)

// Column is one flattened property value destined for page_props.
type Column struct {
	Name  string
	Value any // string, float64, or nil
}

// Relation is one edge extracted from a relation property.
type Relation struct {
	Prop string
	To   string
}

// FlattenProperties converts a page's "properties" object into the columns a
// SQL query sees. The naming follows the official Notion MCP conventions so
// queries written for it keep working here:
//
//   - date properties appear only as "date:Name:start", "date:Name:end" and
//     "date:Name:is_datetime" — never under the bare name;
//   - checkboxes are '__YES__' / '__NO__';
//   - multi-select and other lists are JSON strings;
//   - a property literally named id or url is prefixed with "userDefined:".
//
// Divergences are deliberate and documented in docs/DESIGN.ru.md: rollups
// carry their computed value instead of a rollupResult:// URI, and relations
// carry a JSON array of page ids.
func FlattenProperties(props json.RawMessage) ([]Column, []Relation, error) {
	if len(props) == 0 {
		return nil, nil, nil
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(props, &raw); err != nil {
		return nil, nil, err
	}
	var cols []Column
	var rels []Relation
	for name, value := range raw {
		c, r, err := flattenOne(name, value)
		if err != nil {
			return nil, nil, fmt.Errorf("property %q: %w", name, err)
		}
		cols = append(cols, c...)
		rels = append(rels, r...)
	}
	return cols, rels, nil
}

// ColumnName returns the SQL column name for a property, applying the
// userDefined: escape for names that would collide with the row's own columns.
func ColumnName(property string) string {
	switch strings.ToLower(property) {
	case "id", "url":
		return "userDefined:" + property
	}
	return property
}

// DateColumns returns the three column names a date property produces.
func DateColumns(property string) [3]string {
	return [3]string{
		"date:" + property + ":start",
		"date:" + property + ":end",
		"date:" + property + ":is_datetime",
	}
}

func flattenOne(name string, value json.RawMessage) ([]Column, []Relation, error) {
	var head struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(value, &head); err != nil {
		return nil, nil, err
	}
	col := ColumnName(name)

	switch head.Type {
	case "title", "rich_text":
		var v struct {
			Title    []notion.RichText `json:"title"`
			RichText []notion.RichText `json:"rich_text"`
		}
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		spans := v.Title
		if head.Type == "rich_text" {
			spans = v.RichText
		}
		return []Column{{col, notion.PlainTextOf(spans)}}, nil, nil

	case "number":
		var v struct {
			Number *float64 `json:"number"`
		}
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		if v.Number == nil {
			return []Column{{col, nil}}, nil, nil
		}
		return []Column{{col, *v.Number}}, nil, nil

	case "select", "status":
		var v struct {
			Select *struct {
				Name string `json:"name"`
			} `json:"select"`
			Status *struct {
				Name string `json:"name"`
			} `json:"status"`
		}
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		switch {
		case v.Select != nil:
			return []Column{{col, v.Select.Name}}, nil, nil
		case v.Status != nil:
			return []Column{{col, v.Status.Name}}, nil, nil
		}
		return []Column{{col, nil}}, nil, nil

	case "multi_select":
		var v struct {
			MultiSelect []struct {
				Name string `json:"name"`
			} `json:"multi_select"`
		}
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		names := make([]string, 0, len(v.MultiSelect))
		for _, o := range v.MultiSelect {
			names = append(names, o.Name)
		}
		return []Column{{col, jsonString(names)}}, nil, nil

	case "date":
		var v struct {
			Date *struct {
				Start string `json:"start"`
				End   string `json:"end"`
			} `json:"date"`
		}
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		names := DateColumns(name)
		if v.Date == nil {
			return []Column{{names[0], nil}, {names[1], nil}, {names[2], nil}}, nil, nil
		}
		isDateTime := 0.0
		if strings.Contains(v.Date.Start, "T") {
			isDateTime = 1
		}
		out := []Column{{names[0], v.Date.Start}, {names[2], isDateTime}}
		if v.Date.End != "" {
			out = append(out, Column{names[1], v.Date.End})
		} else {
			out = append(out, Column{names[1], nil})
		}
		return out, nil, nil

	case "checkbox":
		var v struct {
			Checkbox bool `json:"checkbox"`
		}
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		if v.Checkbox {
			return []Column{{col, "__YES__"}}, nil, nil
		}
		return []Column{{col, "__NO__"}}, nil, nil

	case "url", "email", "phone_number":
		var v map[string]any
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		if s, ok := v[head.Type].(string); ok {
			return []Column{{col, s}}, nil, nil
		}
		return []Column{{col, nil}}, nil, nil

	case "people", "created_by", "last_edited_by":
		ids := peopleIDs(value, head.Type)
		return []Column{{col, jsonString(ids)}}, nil, nil

	case "relation":
		var v struct {
			Relation []struct {
				ID string `json:"id"`
			} `json:"relation"`
		}
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		ids := make([]string, 0, len(v.Relation))
		rels := make([]Relation, 0, len(v.Relation))
		for _, r := range v.Relation {
			ids = append(ids, r.ID)
			rels = append(rels, Relation{Prop: name, To: r.ID})
		}
		return []Column{{col, jsonString(ids)}}, rels, nil

	case "files":
		var v struct {
			Files []struct {
				Name     string `json:"name"`
				External struct {
					URL string `json:"url"`
				} `json:"external"`
				File struct {
					URL string `json:"url"`
				} `json:"file"`
			} `json:"files"`
		}
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		out := make([]map[string]string, 0, len(v.Files))
		for _, f := range v.Files {
			url := f.External.URL
			if url == "" {
				url = f.File.URL
			}
			out = append(out, map[string]string{"name": f.Name, "url": url})
		}
		return []Column{{col, jsonString(out)}}, nil, nil

	case "created_time", "last_edited_time":
		var v map[string]any
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		if s, ok := v[head.Type].(string); ok {
			return []Column{{col, s}}, nil, nil
		}
		return []Column{{col, nil}}, nil, nil

	case "unique_id":
		var v struct {
			UniqueID struct {
				Prefix *string `json:"prefix"`
				Number *int64  `json:"number"`
			} `json:"unique_id"`
		}
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		if v.UniqueID.Number == nil {
			return []Column{{col, nil}}, nil, nil
		}
		if v.UniqueID.Prefix != nil && *v.UniqueID.Prefix != "" {
			return []Column{{col, fmt.Sprintf("%s-%d", *v.UniqueID.Prefix, *v.UniqueID.Number)}}, nil, nil
		}
		return []Column{{col, float64(*v.UniqueID.Number)}}, nil, nil

	case "formula":
		var v struct {
			Formula map[string]any `json:"formula"`
		}
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		return []Column{{col, scalarOfTyped(v.Formula)}}, nil, nil

	case "rollup":
		var v struct {
			Rollup map[string]any `json:"rollup"`
		}
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		return []Column{{col, rollupValue(v.Rollup)}}, nil, nil

	case "verification":
		var v struct {
			Verification struct {
				State string `json:"state"`
			} `json:"verification"`
		}
		if err := json.Unmarshal(value, &v); err != nil {
			return nil, nil, err
		}
		return []Column{{col, v.Verification.State}}, nil, nil

	default:
		// Unknown property types keep their raw JSON rather than disappearing.
		return []Column{{col, string(value)}}, nil, nil
	}
}

func peopleIDs(value json.RawMessage, key string) []string {
	var single struct {
		CreatedBy    *struct{ ID string }  `json:"created_by"`
		LastEditedBy *struct{ ID string }  `json:"last_edited_by"`
		People       []struct{ ID string } `json:"people"`
	}
	if err := json.Unmarshal(value, &single); err != nil {
		return nil
	}
	switch key {
	case "created_by":
		if single.CreatedBy != nil {
			return []string{single.CreatedBy.ID}
		}
	case "last_edited_by":
		if single.LastEditedBy != nil {
			return []string{single.LastEditedBy.ID}
		}
	default:
		ids := make([]string, 0, len(single.People))
		for _, p := range single.People {
			ids = append(ids, p.ID)
		}
		return ids
	}
	return nil
}

// scalarOfTyped unwraps Notion's {"type":"number","number":1} shape.
func scalarOfTyped(m map[string]any) any {
	if m == nil {
		return nil
	}
	kind, _ := m["type"].(string)
	switch kind {
	case "number":
		if f, ok := m["number"].(float64); ok {
			return f
		}
	case "string":
		if s, ok := m["string"].(string); ok {
			return s
		}
	case "boolean":
		if b, ok := m["boolean"].(bool); ok {
			if b {
				return "__YES__"
			}
			return "__NO__"
		}
	case "date":
		if d, ok := m["date"].(map[string]any); ok {
			if s, ok := d["start"].(string); ok {
				return s
			}
		}
	}
	return nil
}

// rollupValue flattens a rollup to something comparable in SQL. The official
// server answers with a rollupResult:// URI; a real value is more useful and
// strictly more informative.
func rollupValue(m map[string]any) any {
	if m == nil {
		return nil
	}
	if kind, _ := m["type"].(string); kind == "array" {
		arr, _ := m["array"].([]any)
		parts := make([]any, 0, len(arr))
		for _, item := range arr {
			if mi, ok := item.(map[string]any); ok {
				parts = append(parts, scalarOfTyped(mi))
			}
		}
		return jsonString(parts)
	}
	return scalarOfTyped(m)
}

func jsonString(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "[]"
	}
	return string(b)
}
