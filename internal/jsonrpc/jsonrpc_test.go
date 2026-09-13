package jsonrpc

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestReadRequestAndNotification(t *testing.T) {
	in := strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/list"}` + "\n" +
		`{"jsonrpc":"2.0","method":"notifications/initialized"}` + "\n")
	c := NewConn(in, &bytes.Buffer{})

	first, err := c.Read()
	if err != nil {
		t.Fatalf("read request: %v", err)
	}
	if !first.IsRequest() || first.Method != "tools/list" {
		t.Fatalf("expected a tools/list request, got %+v", first)
	}

	second, err := c.Read()
	if err != nil {
		t.Fatalf("read notification: %v", err)
	}
	if !second.IsNotification() {
		t.Fatalf("expected a notification, got %+v", second)
	}
}

func TestReadLineLongerThanBuffer(t *testing.T) {
	long := strings.Repeat("x", 4<<20)
	payload, err := json.Marshal(&Message{JSONRPC: "2.0", Method: "ping", Params: json.RawMessage(`"` + long + `"`)})
	if err != nil {
		t.Fatal(err)
	}
	c := NewConn(strings.NewReader(string(payload)+"\n"), &bytes.Buffer{})
	m, err := c.Read()
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(m.Params) != len(long)+2 {
		t.Fatalf("params truncated: got %d bytes, want %d", len(m.Params), len(long)+2)
	}
}

func TestRespondWritesOneLine(t *testing.T) {
	var out bytes.Buffer
	c := NewConn(strings.NewReader(""), &out)
	if err := c.Respond(json.RawMessage(`7`), map[string]string{"ok": "yes"}); err != nil {
		t.Fatal(err)
	}
	got := out.String()
	if strings.Count(got, "\n") != 1 || !strings.HasSuffix(got, "\n") {
		t.Fatalf("expected exactly one trailing newline, got %q", got)
	}
	var m Message
	if err := json.Unmarshal([]byte(got), &m); err != nil {
		t.Fatalf("response is not valid json: %v", err)
	}
	if string(m.ID) != "7" || m.Error != nil {
		t.Fatalf("unexpected response %+v", m)
	}
}
