package mirror

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// Two goroutines each read and then write inside one transaction — which is what
// a sync pass and an embedding pass do at the same time on the board.
//
// A deferred transaction starts as a reader and takes the write lock only at its
// first write. SQLite cannot make that upgrade wait: it returns SQLITE_BUSY at
// once, without consulting busy_timeout, so the pass dies rather than queues.
// That is the "database is locked (5)" that killed a data source mid-sync on the
// Pi. Beginning write transactions as IMMEDIATE takes the lock up front, where
// busy_timeout applies and the second writer waits its turn.
func TestConcurrentReadThenWriteTransactionsDoNotFail(t *testing.T) {
	ctx := context.Background()
	db, err := Open(ctx, filepath.Join(t.TempDir(), "mirror.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	const workers = 4
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	start := make(chan struct{})
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			tx, err := db.SQL().BeginTx(ctx, nil)
			if err != nil {
				errs <- err
				return
			}
			defer tx.Rollback()
			// Read first: this is what makes the transaction a reader that has
			// to upgrade later.
			var n int
			if err := tx.QueryRowContext(ctx, `SELECT COUNT(*) FROM sync_log`).Scan(&n); err != nil {
				errs <- err
				return
			}
			time.Sleep(20 * time.Millisecond)
			if _, err := tx.ExecContext(ctx,
				`INSERT INTO sync_log(at,kind,detail) VALUES(?,?,?)`,
				nowISO(), "test", "worker"); err != nil {
				errs <- err
				return
			}
			if err := tx.Commit(); err != nil {
				errs <- err
			}
		}(i)
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("concurrent read-then-write transaction failed: %v", err)
	}

	var n int
	if err := db.SQL().QueryRowContext(ctx, `SELECT COUNT(*) FROM sync_log WHERE kind='test'`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != workers {
		t.Fatalf("committed %d rows, want %d", n, workers)
	}
}
