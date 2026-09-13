// Package ipc connects the stdio MCP process to the sync daemon over a unix
// socket. Only the operations that need the daemon's embedding model or Notion
// token travel this way; SQL and page reads run straight off the mirror file.
package ipc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/index"
	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/jsonrpc"
)

// Handler is what the daemon implements.
type Handler interface {
	Search(ctx context.Context, q index.Query) ([]index.Result, error)
	Resync(ctx context.Context, pageID string, full bool) (string, error)
	Status(ctx context.Context) (map[string]any, error)
}

// Server accepts connections on a unix socket.
type Server struct {
	listener net.Listener
	handler  Handler
	logf     func(format string, args ...any)
}

// Listen creates the socket, replacing a stale one left by a crash.
func Listen(path string, handler Handler, logf func(format string, args ...any)) (*Server, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	if _, err := os.Stat(path); err == nil {
		// A live daemon would be holding it; if a dial fails the socket is dead.
		if conn, dialErr := net.DialTimeout("unix", path, 300*time.Millisecond); dialErr == nil {
			conn.Close()
			return nil, fmt.Errorf("ipc: another daemon is already listening on %s", path)
		}
		if err := os.Remove(path); err != nil {
			return nil, err
		}
	}
	l, err := net.Listen("unix", path)
	if err != nil {
		return nil, err
	}
	if logf == nil {
		logf = func(string, ...any) {}
	}
	return &Server{listener: l, handler: handler, logf: logf}, nil
}

// Addr is the socket path.
func (s *Server) Addr() string { return s.listener.Addr().String() }

// Close stops accepting connections.
func (s *Server) Close() error { return s.listener.Close() }

// Serve handles connections until the listener closes.
func (s *Server) Serve(ctx context.Context) error {
	go func() {
		<-ctx.Done()
		_ = s.listener.Close()
	}()
	for {
		conn, err := s.listener.Accept()
		if err != nil {
			if ctx.Err() != nil || errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		go s.handle(ctx, conn)
	}
}

func (s *Server) handle(ctx context.Context, conn net.Conn) {
	defer conn.Close()
	rpc := jsonrpc.NewConn(conn, conn)
	for {
		msg, err := rpc.Read()
		if err != nil {
			return
		}
		if !msg.IsRequest() {
			continue
		}
		result, rpcErr := s.invoke(ctx, msg.Method, msg.Params)
		if rpcErr != nil {
			_ = rpc.RespondError(msg.ID, rpcErr)
			continue
		}
		_ = rpc.Respond(msg.ID, result)
	}
}

func (s *Server) invoke(ctx context.Context, method string, params json.RawMessage) (any, *jsonrpc.Error) {
	switch method {
	case "search":
		var q index.Query
		if err := json.Unmarshal(params, &q); err != nil {
			return nil, jsonrpc.Errorf(jsonrpc.CodeInvalidParams, "%v", err)
		}
		results, err := s.handler.Search(ctx, q)
		if err != nil {
			return nil, jsonrpc.Errorf(jsonrpc.CodeInternal, "%v", err)
		}
		return map[string]any{"results": results}, nil
	case "resync":
		var p struct {
			PageID string `json:"page_id"`
			Full   bool   `json:"full"`
		}
		if err := json.Unmarshal(params, &p); err != nil {
			return nil, jsonrpc.Errorf(jsonrpc.CodeInvalidParams, "%v", err)
		}
		message, err := s.handler.Resync(ctx, p.PageID, p.Full)
		if err != nil {
			return nil, jsonrpc.Errorf(jsonrpc.CodeInternal, "%v", err)
		}
		return map[string]any{"message": message}, nil
	case "status":
		status, err := s.handler.Status(ctx)
		if err != nil {
			return nil, jsonrpc.Errorf(jsonrpc.CodeInternal, "%v", err)
		}
		return status, nil
	case "ping":
		return map[string]any{"ok": true}, nil
	default:
		return nil, jsonrpc.Errorf(jsonrpc.CodeMethodNotFound, "unknown method %q", method)
	}
}

// Client talks to the daemon. It dials per call: the stdio process makes a
// handful of calls per session, and a fresh connection removes every question
// about a half-dead socket.
type Client struct {
	Path    string
	Timeout time.Duration
}

// NewClient builds a client for a socket path.
func NewClient(path string) *Client {
	return &Client{Path: path, Timeout: 120 * time.Second}
}

// Available reports whether a daemon answers right now.
func (c *Client) Available() bool {
	conn, err := net.DialTimeout("unix", c.Path, 500*time.Millisecond)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

func (c *Client) call(ctx context.Context, method string, params any, into any) error {
	conn, err := net.DialTimeout("unix", c.Path, 2*time.Second)
	if err != nil {
		return fmt.Errorf("ipc: dial %s: %w", c.Path, err)
	}
	defer conn.Close()
	deadline := time.Now().Add(c.Timeout)
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}
	_ = conn.SetDeadline(deadline)

	rpc := jsonrpc.NewConn(conn, conn)
	raw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	if err := rpc.Write(&jsonrpc.Message{ID: json.RawMessage(`1`), Method: method, Params: raw}); err != nil {
		return err
	}
	reply, err := rpc.Read()
	if err != nil {
		return err
	}
	if reply.Error != nil {
		return reply.Error
	}
	if into == nil {
		return nil
	}
	return json.Unmarshal(reply.Result, into)
}

// Search asks the daemon to run a query with its embedding model.
func (c *Client) Search(ctx context.Context, q index.Query) ([]index.Result, error) {
	var out struct {
		Results []index.Result `json:"results"`
	}
	if err := c.call(ctx, "search", q, &out); err != nil {
		return nil, err
	}
	return out.Results, nil
}

// Resync asks the daemon to pull from Notion now.
func (c *Client) Resync(ctx context.Context, pageID string, full bool) (string, error) {
	var out struct {
		Message string `json:"message"`
	}
	params := map[string]any{"page_id": pageID, "full": full}
	if err := c.call(ctx, "resync", params, &out); err != nil {
		return "", err
	}
	return out.Message, nil
}

// Status asks the daemon for its own view of the mirror.
func (c *Client) Status(ctx context.Context) (map[string]any, error) {
	out := map[string]any{}
	if err := c.call(ctx, "status", map[string]any{}, &out); err != nil {
		return nil, err
	}
	return out, nil
}
