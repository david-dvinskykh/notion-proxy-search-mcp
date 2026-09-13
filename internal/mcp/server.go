// Package mcp is a minimal Model Context Protocol server over stdio: the
// handshake, tools/list, tools/call and nothing else. A gateway spawns this
// process per session, so the surface is deliberately small.
package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"sort"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/jsonrpc"
)

// ProtocolVersion is the revision this server implements.
const ProtocolVersion = "2025-06-18"

// supported lists the revisions the server will agree to, newest first.
var supported = []string{"2025-06-18", "2025-03-26", "2024-11-05"}

// Tool is one callable tool.
type Tool struct {
	Name        string
	Title       string
	Description string
	InputSchema json.RawMessage
	// Handler returns either a string (sent as text) or any value, which is
	// marshalled to indented JSON.
	Handler func(ctx context.Context, args json.RawMessage) (any, error)
}

// Server holds the tool registry.
type Server struct {
	Name    string
	Version string
	tools   map[string]Tool
	Logger  *log.Logger
}

// NewServer builds an empty server.
func NewServer(name, version string, logger *log.Logger) *Server {
	return &Server{Name: name, Version: version, tools: map[string]Tool{}, Logger: logger}
}

// Register adds a tool.
func (s *Server) Register(t Tool) { s.tools[t.Name] = t }

// Serve runs the message loop until the peer closes the connection.
func (s *Server) Serve(ctx context.Context, r io.Reader, w io.Writer) error {
	conn := jsonrpc.NewConn(r, w)
	for {
		msg, err := conn.Read()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			var rpcErr *jsonrpc.Error
			if errors.As(err, &rpcErr) {
				// A malformed line gets an error and the loop continues: a
				// single bad frame must not take down the session.
				s.logf("parse error: %v", rpcErr)
				continue
			}
			return err
		}
		if msg.IsNotification() {
			continue
		}
		if !msg.IsRequest() {
			continue
		}
		s.dispatch(ctx, conn, msg)
	}
}

func (s *Server) dispatch(ctx context.Context, conn *jsonrpc.Conn, msg *jsonrpc.Message) {
	switch msg.Method {
	case "initialize":
		_ = conn.Respond(msg.ID, s.initialize(msg.Params))
	case "ping":
		_ = conn.Respond(msg.ID, map[string]any{})
	case "tools/list":
		_ = conn.Respond(msg.ID, map[string]any{"tools": s.list()})
	case "tools/call":
		s.call(ctx, conn, msg)
	case "resources/list":
		_ = conn.Respond(msg.ID, map[string]any{"resources": []any{}})
	case "prompts/list":
		_ = conn.Respond(msg.ID, map[string]any{"prompts": []any{}})
	default:
		_ = conn.RespondError(msg.ID, jsonrpc.Errorf(jsonrpc.CodeMethodNotFound, "unknown method %q", msg.Method))
	}
}

func (s *Server) initialize(params json.RawMessage) map[string]any {
	requested := ""
	if len(params) > 0 {
		var p struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		if err := json.Unmarshal(params, &p); err == nil {
			requested = p.ProtocolVersion
		}
	}
	version := ProtocolVersion
	for _, v := range supported {
		if v == requested {
			version = requested
			break
		}
	}
	return map[string]any{
		"protocolVersion": version,
		"capabilities":    map[string]any{"tools": map[string]any{}},
		"serverInfo":      map[string]any{"name": s.Name, "version": s.Version},
	}
}

type toolDescriptor struct {
	Name        string          `json:"name"`
	Title       string          `json:"title,omitempty"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

func (s *Server) list() []toolDescriptor {
	out := make([]toolDescriptor, 0, len(s.tools))
	for _, t := range s.tools {
		schema := t.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out = append(out, toolDescriptor{
			Name: t.Name, Title: t.Title, Description: t.Description, InputSchema: schema,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (s *Server) call(ctx context.Context, conn *jsonrpc.Conn, msg *jsonrpc.Message) {
	var params struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		_ = conn.RespondError(msg.ID, jsonrpc.Errorf(jsonrpc.CodeInvalidParams, "%v", err))
		return
	}
	tool, ok := s.tools[params.Name]
	if !ok {
		_ = conn.RespondError(msg.ID, jsonrpc.Errorf(jsonrpc.CodeInvalidParams, "unknown tool %q", params.Name))
		return
	}

	result, err := tool.Handler(ctx, params.Arguments)
	if err != nil {
		// A tool failure is reported inside the result, not as a protocol
		// error: the model has to see what went wrong to adapt.
		s.logf("tool %s: %v", params.Name, err)
		_ = conn.Respond(msg.ID, errorResult(err))
		return
	}
	_ = conn.Respond(msg.ID, textResult(result))
}

// TextResult wraps a value as MCP tool content.
func textResult(value any) map[string]any {
	var text string
	switch v := value.(type) {
	case string:
		text = v
	case nil:
		text = ""
	default:
		encoded, err := json.MarshalIndent(v, "", "  ")
		if err != nil {
			return errorResult(err)
		}
		text = string(encoded)
	}
	return map[string]any{
		"content": []any{map[string]any{"type": "text", "text": text}},
	}
}

func errorResult(err error) map[string]any {
	return map[string]any{
		"content": []any{map[string]any{"type": "text", "text": err.Error()}},
		"isError": true,
	}
}

func (s *Server) logf(format string, args ...any) {
	if s.Logger != nil {
		s.Logger.Printf(format, args...)
	}
}

// Describe renders the registered tools as a human-readable list, used by the
// CLI's own help output.
func (s *Server) Describe() string {
	out := ""
	for _, t := range s.list() {
		out += fmt.Sprintf("%-28s %s\n", t.Name, t.Description)
	}
	return out
}
