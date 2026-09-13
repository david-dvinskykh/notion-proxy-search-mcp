package ipc

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/david-dvinskykh/notion-proxy-search-mcp/internal/index"
)

type fakeHandler struct {
	lastQuery index.Query
	fail      bool
}

func (f *fakeHandler) Search(_ context.Context, q index.Query) ([]index.Result, error) {
	f.lastQuery = q
	if f.fail {
		return nil, fmt.Errorf("model not loaded")
	}
	return []index.Result{{PageID: "p1", Title: "Факт", Score: 0.5, Via: "both"}}, nil
}

func (f *fakeHandler) Resync(_ context.Context, pageID string, full bool) (string, error) {
	return fmt.Sprintf("resynced %s full=%v", pageID, full), nil
}

func (f *fakeHandler) Status(context.Context) (map[string]any, error) {
	return map[string]any{"counts": map[string]any{"pages": 2}}, nil
}

func startServer(t *testing.T, h Handler) *Client {
	t.Helper()
	path := filepath.Join(t.TempDir(), "daemon.sock")
	srv, err := Listen(path, h, t.Logf)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() {
		cancel()
		srv.Close()
	})
	go func() { _ = srv.Serve(ctx) }()
	return NewClient(path)
}

func TestSearchRoundTripPreservesQueryFields(t *testing.T) {
	h := &fakeHandler{}
	client := startServer(t, h)

	want := index.Query{
		Text: "сколько памяти", Limit: 5, Mode: index.ModeHybrid,
		DataSources: []string{"ds-facts"}, EditedAfter: "2026-09-01", ExpandRelations: true,
	}
	got, err := client.Search(context.Background(), want)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].PageID != "p1" || got[0].Via != "both" {
		t.Fatalf("results = %+v", got)
	}
	if h.lastQuery.Text != want.Text || h.lastQuery.Limit != want.Limit ||
		h.lastQuery.Mode != want.Mode || !h.lastQuery.ExpandRelations ||
		h.lastQuery.EditedAfter != want.EditedAfter ||
		len(h.lastQuery.DataSources) != 1 || h.lastQuery.DataSources[0] != "ds-facts" {
		t.Fatalf("query lost fields in transit: %+v", h.lastQuery)
	}
}

func TestHandlerErrorReachesTheClient(t *testing.T) {
	client := startServer(t, &fakeHandler{fail: true})
	if _, err := client.Search(context.Background(), index.Query{Text: "x"}); err == nil {
		t.Fatal("expected the handler error to surface")
	}
}

func TestResyncAndStatus(t *testing.T) {
	client := startServer(t, &fakeHandler{})
	msg, err := client.Resync(context.Background(), "p1", true)
	if err != nil {
		t.Fatal(err)
	}
	if msg != "resynced p1 full=true" {
		t.Fatalf("message = %q", msg)
	}
	status, err := client.Status(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if status["counts"] == nil {
		t.Fatalf("status = %+v", status)
	}
}

func TestAvailableIsFalseWithoutSocket(t *testing.T) {
	client := NewClient(filepath.Join(t.TempDir(), "absent.sock"))
	if client.Available() {
		t.Fatal("a missing socket must not look available")
	}
	if _, err := client.Search(context.Background(), index.Query{Text: "x"}); err == nil {
		t.Fatal("expected a dial error")
	}
}

func TestListenReplacesStaleSocketFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "daemon.sock")
	first, err := Listen(path, &fakeHandler{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Close the listener without removing the file, the way a crash leaves it.
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := Listen(path, &fakeHandler{}, nil)
	if err != nil {
		t.Fatalf("a stale socket must be replaced: %v", err)
	}
	second.Close()
}

func TestListenRefusesASecondLiveDaemon(t *testing.T) {
	path := filepath.Join(t.TempDir(), "daemon.sock")
	first, err := Listen(path, &fakeHandler{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = first.Serve(ctx) }()

	if _, err := Listen(path, &fakeHandler{}, nil); err == nil {
		t.Fatal("two daemons must not share one socket")
	}
}
