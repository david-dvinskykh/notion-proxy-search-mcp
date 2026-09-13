package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"
)

// session drives a server over a pipe and collects its replies.
func session(t *testing.T, s *Server, requests ...string) []map[string]any {
	t.Helper()
	in := strings.NewReader(strings.Join(requests, "\n") + "\n")
	var out strings.Builder
	if err := s.Serve(context.Background(), in, &out); err != nil && !errors.Is(err, io.EOF) {
		t.Fatalf("serve: %v", err)
	}
	var replies []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(out.String()), "\n") {
		if line == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("reply %q is not json: %v", line, err)
		}
		replies = append(replies, m)
	}
	return replies
}

func echoServer() *Server {
	s := NewServer("test", "0.0.1", nil)
	s.Register(Tool{
		Name:        "echo",
		Description: "returns its argument",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}`),
		Handler: func(ctx context.Context, args json.RawMessage) (any, error) {
			var p struct{ Text string }
			if err := json.Unmarshal(args, &p); err != nil {
				return nil, err
			}
			if p.Text == "boom" {
				return nil, fmt.Errorf("asked to fail")
			}
			return p.Text, nil
		},
	})
	return s
}

func TestInitializeEchoesSupportedProtocolVersion(t *testing.T) {
	replies := session(t, echoServer(),
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{}}}`)
	if len(replies) != 1 {
		t.Fatalf("got %d replies", len(replies))
	}
	result := replies[0]["result"].(map[string]any)
	if result["protocolVersion"] != "2025-03-26" {
		t.Fatalf("protocolVersion = %v, want the client's own supported value", result["protocolVersion"])
	}
	caps := result["capabilities"].(map[string]any)
	if _, ok := caps["tools"]; !ok {
		t.Fatalf("tools capability missing: %+v", caps)
	}
}

func TestInitializeFallsBackForUnknownVersion(t *testing.T) {
	replies := session(t, echoServer(),
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1999-01-01"}}`)
	result := replies[0]["result"].(map[string]any)
	if result["protocolVersion"] != ProtocolVersion {
		t.Fatalf("protocolVersion = %v, want %s", result["protocolVersion"], ProtocolVersion)
	}
}

func TestToolsListAndCall(t *testing.T) {
	replies := session(t, echoServer(),
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		`{"jsonrpc":"2.0","id":2,"method":"tools/list"}`,
		`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"text":"привет"}}}`)
	if len(replies) != 2 {
		t.Fatalf("a notification must not be answered; got %d replies", len(replies))
	}

	tools := replies[0]["result"].(map[string]any)["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["name"] != "echo" {
		t.Fatalf("tools = %+v", tools)
	}
	if _, ok := tools[0].(map[string]any)["inputSchema"]; !ok {
		t.Fatal("inputSchema is required by the protocol")
	}

	content := replies[1]["result"].(map[string]any)["content"].([]any)
	first := content[0].(map[string]any)
	if first["type"] != "text" || first["text"] != "привет" {
		t.Fatalf("content = %+v", content)
	}
}

func TestToolFailureIsReportedInsideTheResult(t *testing.T) {
	replies := session(t, echoServer(),
		`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"text":"boom"}}}`)
	result := replies[0]["result"].(map[string]any)
	if result["isError"] != true {
		t.Fatalf("a failing tool must set isError: %+v", result)
	}
	if replies[0]["error"] != nil {
		t.Fatal("a tool failure must not become a protocol error")
	}
}

func TestUnknownMethodAndUnknownTool(t *testing.T) {
	replies := session(t, echoServer(),
		`{"jsonrpc":"2.0","id":1,"method":"does/not/exist"}`,
		`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"nope","arguments":{}}}`)
	if replies[0]["error"] == nil {
		t.Fatal("unknown method must be a protocol error")
	}
	if replies[1]["error"] == nil {
		t.Fatal("unknown tool must be a protocol error")
	}
}

func TestPingAnswersEmptyResult(t *testing.T) {
	replies := session(t, echoServer(), `{"jsonrpc":"2.0","id":1,"method":"ping"}`)
	if _, ok := replies[0]["result"].(map[string]any); !ok {
		t.Fatalf("ping reply = %+v", replies[0])
	}
}

func TestStructuredResultIsJSONEncoded(t *testing.T) {
	s := NewServer("test", "0", nil)
	s.Register(Tool{Name: "rows", Handler: func(context.Context, json.RawMessage) (any, error) {
		return map[string]any{"n": 2}, nil
	}})
	replies := session(t, s, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"rows","arguments":{}}}`)
	text := replies[0]["result"].(map[string]any)["content"].([]any)[0].(map[string]any)["text"].(string)
	var decoded map[string]any
	if err := json.Unmarshal([]byte(text), &decoded); err != nil {
		t.Fatalf("structured result is not valid json: %v", err)
	}
	if decoded["n"] != 2.0 {
		t.Fatalf("decoded = %+v", decoded)
	}
}
