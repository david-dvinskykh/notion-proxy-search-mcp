package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/mcp"
)

// Register publishes the tool set on an MCP server. The names mirror the
// official Notion MCP tools an agent already knows, so the proxy can stand in
// for the calls that run into the hosted server's quotas.
func (s *Service) Register(srv *mcp.Server) {
	srv.Register(mcp.Tool{
		Name:  "notion-search",
		Title: "Search the mirrored Notion workspace",
		Description: "Hybrid search over a local mirror of Notion: embeddings plus BM25, fused, " +
			"with optional one-hop relation expansion. No Notion API quota is consumed. " +
			"Use mode=keyword for exact strings, mode=vector for paraphrases, hybrid (default) otherwise.",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "Natural language question or keywords."},
    "limit": {"type": "integer", "description": "Maximum pages to return (default 10)."},
    "mode": {"type": "string", "enum": ["hybrid", "vector", "keyword"], "description": "Retrieval branches to run. Default hybrid."},
    "data_sources": {"type": "array", "items": {"type": "string"}, "description": "Restrict to these data source ids."},
    "page_ids": {"type": "array", "items": {"type": "string"}, "description": "Restrict to these page ids."},
    "edited_after": {"type": "string", "description": "ISO 8601 lower bound on last_edited_time."},
    "edited_before": {"type": "string", "description": "ISO 8601 upper bound on last_edited_time."},
    "expand_relations": {"type": "boolean", "description": "Also return pages one relation hop away, ranked below direct hits."}
  },
  "required": ["query"]
}`),
		Handler: func(ctx context.Context, raw json.RawMessage) (any, error) {
			var args SearchArgs
			if err := decode(raw, &args); err != nil {
				return nil, err
			}
			return s.Search(ctx, args)
		},
	})

	srv.Register(mcp.Tool{
		Name:        "notion-fetch",
		Title:       "Read one mirrored page",
		Description: "Return a mirrored Notion page as Markdown with its properties, by page id or URL. Served from the local mirror, so it never hits a rate limit.",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "id": {"type": "string", "description": "Page id, dashed uuid, or any Notion URL."},
    "include_relations": {"type": "boolean", "description": "List the pages this one points at."}
  },
  "required": ["id"]
}`),
		Handler: func(ctx context.Context, raw json.RawMessage) (any, error) {
			var args FetchArgs
			if err := decode(raw, &args); err != nil {
				return nil, err
			}
			return s.Fetch(ctx, args)
		},
	})

	srv.Register(mcp.Tool{
		Name:  "notion-query-data-sources",
		Title: "SQL over mirrored data sources",
		Description: "Run read-only SQLite SQL against the mirror. Every Notion data source is a table named " +
			"\"collection://<data source id>\", with the same column conventions as the official server: date " +
			"properties appear only as \"date:Name:start\" / \":end\" / \":is_datetime\", checkboxes as '__YES__' / " +
			"'__NO__', multi-selects and relations as JSON arrays. Several data sources can be joined in one " +
			"statement, and there is no per-workspace query quota. Extra columns every table carries: id, url, " +
			"title, icon, created_time, last_edited_time, content_markdown, parent_id.",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "A single SELECT or WITH statement."},
    "params": {"type": "array", "description": "Positional parameters bound to ? placeholders."},
    "limit": {"type": "integer", "description": "Maximum rows to return (default and cap 500)."}
  },
  "required": ["query"]
}`),
		Handler: func(ctx context.Context, raw json.RawMessage) (any, error) {
			var args QueryArgs
			if err := decode(raw, &args); err != nil {
				return nil, err
			}
			return s.Query(ctx, args)
		},
	})

	srv.Register(mcp.Tool{
		Name:        "notion-list-data-sources",
		Title:       "Catalog of mirrored data sources",
		Description: "List every mirrored data source with its table name, column names and row count — the schema needed to write a query without fetching each database first.",
		InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
		Handler: func(ctx context.Context, _ json.RawMessage) (any, error) {
			return s.ListDataSources(ctx)
		},
	})

	srv.Register(mcp.Tool{
		Name:  "notion-recall",
		Title: "Everything known about one entity",
		Description: "Find the page that best matches a name, then return every page linked to it, grouped by the " +
			"database they live in. One call in place of a search followed by a fetch per relation.",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "entity": {"type": "string", "description": "Person, organisation, project or any page title."},
    "limit": {"type": "integer", "description": "How many candidate matches to consider (default 8)."}
  },
  "required": ["entity"]
}`),
		Handler: func(ctx context.Context, raw json.RawMessage) (any, error) {
			var args RecallArgs
			if err := decode(raw, &args); err != nil {
				return nil, err
			}
			return s.Recall(ctx, args)
		},
	})

	srv.Register(mcp.Tool{
		Name:        "notion-mirror-status",
		Title:       "Mirror freshness and size",
		Description: "Report mirrored page and chunk counts, embedding backlog, sync watermarks and the last sync log lines.",
		InputSchema: json.RawMessage(`{"type": "object", "properties": {}}`),
		Handler: func(ctx context.Context, _ json.RawMessage) (any, error) {
			return s.Status(ctx)
		},
	})

	srv.Register(mcp.Tool{
		Name:  "notion-resync",
		Title: "Pull fresh data from Notion now",
		Description: "Re-read one page, or run a whole sync pass, instead of waiting for the next poll. Use it right " +
			"after writing to Notion through the official server when the next read must see the change.",
		InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "page_id": {"type": "string", "description": "Page id or URL to re-read. Omit to run a workspace pass."},
    "full": {"type": "boolean", "description": "Ignore watermarks and re-read every row, also detecting deletions."}
  }
}`),
		Handler: func(ctx context.Context, raw json.RawMessage) (any, error) {
			var args ResyncArgs
			if err := decode(raw, &args); err != nil {
				return nil, err
			}
			return s.Resync(ctx, args)
		},
	})
}

// decode parses tool arguments, treating an absent object as empty.
func decode(raw json.RawMessage, into any) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	if err := json.Unmarshal(raw, into); err != nil {
		return fmt.Errorf("invalid arguments: %w", err)
	}
	return nil
}
