package notion

import (
	"context"
	"encoding/json"
	"net/url"
	"strings"
)

// Object is the common envelope Notion puts on every entity.
type Object struct {
	Object         string          `json:"object"`
	ID             string          `json:"id"`
	CreatedTime    string          `json:"created_time"`
	LastEditedTime string          `json:"last_edited_time"`
	Archived       bool            `json:"archived"`
	InTrash        bool            `json:"in_trash"`
	URL            string          `json:"url"`
	Icon           json.RawMessage `json:"icon"`
	Cover          json.RawMessage `json:"cover"`
	Parent         Parent          `json:"parent"`
	Properties     json.RawMessage `json:"properties"`
	Title          []RichText      `json:"title"`
	DatabaseParent string          `json:"-"`
	Raw            json.RawMessage `json:"-"`
}

// Parent identifies where an object lives.
type Parent struct {
	Type         string `json:"type"`
	PageID       string `json:"page_id"`
	DatabaseID   string `json:"database_id"`
	DataSourceID string `json:"data_source_id"`
	BlockID      string `json:"block_id"`
	Workspace    bool   `json:"workspace"`
}

// IsDatabaseRow reports whether the object is a row of a database rather than a
// standalone page. The type string is deliberately not consulted: it is
// "database_id" in API 2022-06-28 and "data_source_id" in 2025-09-03, and a
// mismatch there silently turns every row into a loose page.
func (p Parent) IsDatabaseRow() bool {
	return p.DataSourceID != "" || p.DatabaseID != ""
}

// RowDataSource returns the data source a row belongs to, preferring the data
// source id and falling back to the database id for older payloads. It is empty
// for anything that is not a database row.
func (p Parent) RowDataSource() string {
	if p.DataSourceID != "" {
		return p.DataSourceID
	}
	return p.DatabaseID
}

// ID returns the parent's identifier regardless of its type.
func (p Parent) ParentID() string {
	switch {
	case p.PageID != "":
		return p.PageID
	case p.DataSourceID != "":
		return p.DataSourceID
	case p.DatabaseID != "":
		return p.DatabaseID
	case p.BlockID != "":
		return p.BlockID
	default:
		return ""
	}
}

// list is the shape of every paginated Notion response.
type list struct {
	Results    []json.RawMessage `json:"results"`
	NextCursor *string           `json:"next_cursor"`
	HasMore    bool              `json:"has_more"`
}

// Page wraps a page or database row with its raw JSON kept for the mirror.
type Page struct {
	Object
}

// UnmarshalJSON keeps the untouched payload alongside the parsed fields so the
// mirror can store properties without modelling every property type here.
func (p *Page) UnmarshalJSON(data []byte) error {
	type alias Object
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	p.Object = Object(a)
	p.Raw = append(json.RawMessage(nil), data...)
	return nil
}

// SearchParams controls a workspace search call.
type SearchParams struct {
	Query      string
	ObjectType string // "page", "database" or "data_source"; empty means all
	Ascending  bool
	PageSize   int
}

// Search walks every page of /v1/search and calls visit for each result. The
// walk stops early when visit returns false.
func (c *Client) Search(ctx context.Context, p SearchParams, visit func(raw json.RawMessage) bool) error {
	pageSize := p.PageSize
	if pageSize <= 0 {
		pageSize = 100
	}
	direction := "descending"
	if p.Ascending {
		direction = "ascending"
	}
	var cursor *string
	for {
		body := map[string]any{
			"page_size": pageSize,
			"sort":      map[string]any{"direction": direction, "timestamp": "last_edited_time"},
		}
		if p.Query != "" {
			body["query"] = p.Query
		}
		if p.ObjectType != "" {
			body["filter"] = map[string]any{"property": "object", "value": p.ObjectType}
		}
		if cursor != nil {
			body["start_cursor"] = *cursor
		}
		raw, err := c.do(ctx, "POST", "/search", body)
		if err != nil {
			return err
		}
		var l list
		if err := json.Unmarshal(raw, &l); err != nil {
			return err
		}
		for _, r := range l.Results {
			if !visit(r) {
				return nil
			}
		}
		if !l.HasMore || l.NextCursor == nil {
			return nil
		}
		cursor = l.NextCursor
	}
}

// QueryParams controls a data source query.
type QueryParams struct {
	Filter   map[string]any
	Sorts    []map[string]any
	PageSize int
}

// QueryDataSource walks every page of a data source query. Unlike search, this
// endpoint is strongly consistent, which is why incremental sync prefers it.
func (c *Client) QueryDataSource(ctx context.Context, dataSourceID string, p QueryParams, visit func(raw json.RawMessage) bool) error {
	pageSize := p.PageSize
	if pageSize <= 0 {
		pageSize = 100
	}
	var cursor *string
	for {
		body := map[string]any{"page_size": pageSize}
		if p.Filter != nil {
			body["filter"] = p.Filter
		}
		if len(p.Sorts) > 0 {
			body["sorts"] = p.Sorts
		}
		if cursor != nil {
			body["start_cursor"] = *cursor
		}
		raw, err := c.do(ctx, "POST", "/data_sources/"+clean(dataSourceID)+"/query", body)
		if err != nil {
			return err
		}
		var l list
		if err := json.Unmarshal(raw, &l); err != nil {
			return err
		}
		for _, r := range l.Results {
			if !visit(r) {
				return nil
			}
		}
		if !l.HasMore || l.NextCursor == nil {
			return nil
		}
		cursor = l.NextCursor
	}
}

// EditedAfterFilter builds the timestamp filter used for incremental syncs.
func EditedAfterFilter(iso string) map[string]any {
	return map[string]any{
		"timestamp":        "last_edited_time",
		"last_edited_time": map[string]any{"on_or_after": iso},
	}
}

// AscendingEditedSort orders rows oldest-edit-first so an interrupted sync can
// resume from the last row it stored.
func AscendingEditedSort() []map[string]any {
	return []map[string]any{{"timestamp": "last_edited_time", "direction": "ascending"}}
}

// Database is a database container with its data sources.
type Database struct {
	Object
	DataSources []struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	} `json:"data_sources"`
}

// RetrieveDatabase fetches a database container.
func (c *Client) RetrieveDatabase(ctx context.Context, id string) (*Database, error) {
	raw, err := c.do(ctx, "GET", "/databases/"+clean(id), nil)
	if err != nil {
		return nil, err
	}
	var db Database
	if err := json.Unmarshal(raw, &db); err != nil {
		return nil, err
	}
	db.Raw = raw
	return &db, nil
}

// DataSource is one schema under a database.
type DataSource struct {
	Object
	DatabaseParentID string          `json:"-"`
	Schema           json.RawMessage `json:"properties"`
}

// RetrieveDataSource fetches a data source with its property schema.
func (c *Client) RetrieveDataSource(ctx context.Context, id string) (*DataSource, error) {
	raw, err := c.do(ctx, "GET", "/data_sources/"+clean(id), nil)
	if err != nil {
		return nil, err
	}
	var ds DataSource
	if err := json.Unmarshal(raw, &ds); err != nil {
		return nil, err
	}
	ds.Raw = raw
	ds.DatabaseParentID = ds.Parent.DatabaseID
	return &ds, nil
}

// RetrievePage fetches a single page with its properties.
func (c *Client) RetrievePage(ctx context.Context, id string) (*Page, error) {
	raw, err := c.do(ctx, "GET", "/pages/"+clean(id), nil)
	if err != nil {
		return nil, err
	}
	var p Page
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, err
	}
	return &p, nil
}

// Block is one content block. Children are fetched separately.
type Block struct {
	Object
	Type        string          `json:"type"`
	HasChildren bool            `json:"has_children"`
	Raw         json.RawMessage `json:"-"`
}

// UnmarshalJSON keeps the raw block so the renderer can reach type-specific
// fields without a struct per block type.
func (b *Block) UnmarshalJSON(data []byte) error {
	type alias struct {
		Object
		Type        string `json:"type"`
		HasChildren bool   `json:"has_children"`
	}
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	b.Object = a.Object
	b.Type = a.Type
	b.HasChildren = a.HasChildren
	b.Raw = append(json.RawMessage(nil), data...)
	return nil
}

// BlockChildren returns the direct children of a block or page.
func (c *Client) BlockChildren(ctx context.Context, id string) ([]Block, error) {
	var out []Block
	var cursor *string
	for {
		q := url.Values{"page_size": []string{"100"}}
		if cursor != nil {
			q.Set("start_cursor", *cursor)
		}
		raw, err := c.do(ctx, "GET", "/blocks/"+clean(id)+"/children?"+q.Encode(), nil)
		if err != nil {
			return nil, err
		}
		var l list
		if err := json.Unmarshal(raw, &l); err != nil {
			return nil, err
		}
		for _, r := range l.Results {
			var b Block
			if err := json.Unmarshal(r, &b); err != nil {
				return nil, err
			}
			out = append(out, b)
		}
		if !l.HasMore || l.NextCursor == nil {
			return out, nil
		}
		cursor = l.NextCursor
	}
}

// User is a workspace member or bot.
type User struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Type      string `json:"type"`
	AvatarURL string `json:"avatar_url"`
	Person    struct {
		Email string `json:"email"`
	} `json:"person"`
}

// Users lists every user the integration can see.
func (c *Client) Users(ctx context.Context) ([]User, error) {
	var out []User
	var cursor *string
	for {
		q := url.Values{"page_size": []string{"100"}}
		if cursor != nil {
			q.Set("start_cursor", *cursor)
		}
		raw, err := c.do(ctx, "GET", "/users?"+q.Encode(), nil)
		if err != nil {
			return nil, err
		}
		var l list
		if err := json.Unmarshal(raw, &l); err != nil {
			return nil, err
		}
		for _, r := range l.Results {
			var u User
			if err := json.Unmarshal(r, &u); err != nil {
				return nil, err
			}
			out = append(out, u)
		}
		if !l.HasMore || l.NextCursor == nil {
			return out, nil
		}
		cursor = l.NextCursor
	}
}

// clean normalizes an id or URL into the dashless-or-dashed id Notion accepts.
func clean(id string) string {
	id = strings.TrimSpace(id)
	if i := strings.LastIndex(id, "/"); i >= 0 {
		id = id[i+1:]
	}
	if i := strings.Index(id, "?"); i >= 0 {
		id = id[:i]
	}
	if i := strings.LastIndex(id, "-"); i > 0 && len(id)-i-1 == 32 {
		id = id[i+1:]
	}
	return id
}
