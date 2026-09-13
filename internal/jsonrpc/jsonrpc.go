// Package jsonrpc implements the subset of JSON-RPC 2.0 that both the MCP
// stdio transport and the daemon's unix socket need: newline-delimited
// messages, requests with ids, notifications without them, and errors.
package jsonrpc

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sync"
)

// Standard JSON-RPC error codes plus the ones MCP relies on.
const (
	CodeParse          = -32700
	CodeInvalidRequest = -32600
	CodeMethodNotFound = -32601
	CodeInvalidParams  = -32602
	CodeInternal       = -32603
)

// Message is a single JSON-RPC frame. Requests, responses and notifications
// share one struct so a reader does not have to guess the shape up front.
type Message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *Error          `json:"error,omitempty"`
}

// IsRequest reports whether the message expects a response.
func (m *Message) IsRequest() bool { return m.Method != "" && len(m.ID) > 0 }

// IsNotification reports whether the message is a fire-and-forget call.
func (m *Message) IsNotification() bool { return m.Method != "" && len(m.ID) == 0 }

// Error is a JSON-RPC error object.
type Error struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *Error) Error() string { return fmt.Sprintf("jsonrpc %d: %s", e.Code, e.Message) }

// Errorf builds an error object with a formatted message.
func Errorf(code int, format string, args ...any) *Error {
	return &Error{Code: code, Message: fmt.Sprintf(format, args...)}
}

// Conn is a line-delimited JSON-RPC connection. Writes are serialized so
// concurrent handlers cannot interleave halves of two frames.
type Conn struct {
	r  *bufio.Reader
	w  io.Writer
	mu sync.Mutex
}

// NewConn wraps a reader/writer pair, typically stdin/stdout or a socket.
func NewConn(r io.Reader, w io.Writer) *Conn {
	// Notion markdown payloads routinely exceed the default 64 KiB line cap.
	return &Conn{r: bufio.NewReaderSize(r, 1<<20), w: w}
}

// Read returns the next frame, or io.EOF when the peer is gone. A malformed
// line yields a parse error without closing the connection.
func (c *Conn) Read() (*Message, error) {
	for {
		line, err := c.readLine()
		if err != nil {
			return nil, err
		}
		if len(line) == 0 {
			continue
		}
		var m Message
		if err := json.Unmarshal(line, &m); err != nil {
			return nil, &Error{Code: CodeParse, Message: err.Error()}
		}
		return &m, nil
	}
}

// readLine reads one logical line regardless of its length.
func (c *Conn) readLine() ([]byte, error) {
	var buf []byte
	for {
		chunk, err := c.r.ReadSlice('\n')
		buf = append(buf, chunk...)
		if err == nil {
			return trimEOL(buf), nil
		}
		if err == bufio.ErrBufferFull {
			continue
		}
		if err == io.EOF && len(trimEOL(buf)) > 0 {
			return trimEOL(buf), nil
		}
		return nil, err
	}
}

func trimEOL(b []byte) []byte {
	for len(b) > 0 && (b[len(b)-1] == '\n' || b[len(b)-1] == '\r') {
		b = b[:len(b)-1]
	}
	return b
}

// Write sends one frame followed by a newline.
func (c *Conn) Write(m *Message) error {
	m.JSONRPC = "2.0"
	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	c.mu.Lock()
	defer c.mu.Unlock()
	_, err = c.w.Write(data)
	return err
}

// Respond replies to a request with a result value.
func (c *Conn) Respond(id json.RawMessage, result any) error {
	raw, err := json.Marshal(result)
	if err != nil {
		return c.RespondError(id, &Error{Code: CodeInternal, Message: err.Error()})
	}
	return c.Write(&Message{ID: id, Result: raw})
}

// RespondError replies to a request with an error object.
func (c *Conn) RespondError(id json.RawMessage, e *Error) error {
	return c.Write(&Message{ID: id, Error: e})
}

// Notify sends a notification.
func (c *Conn) Notify(method string, params any) error {
	raw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	return c.Write(&Message{Method: method, Params: raw})
}
