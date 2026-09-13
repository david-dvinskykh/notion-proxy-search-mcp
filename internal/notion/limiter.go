package notion

import (
	"context"
	"sync"
	"time"
)

// limiter is a single-token-at-a-time pacer: it lets one request through per
// interval. A token bucket with burst would be faster but Notion counts an
// average, and a burst at the start of a crawl is exactly what trips it.
type limiter struct {
	mu       sync.Mutex
	interval time.Duration
	next     time.Time
}

func newLimiter(rps float64) *limiter {
	return &limiter{interval: time.Duration(float64(time.Second) / rps)}
}

// wait blocks until the caller may issue a request.
func (l *limiter) wait(ctx context.Context) error {
	l.mu.Lock()
	now := time.Now()
	if l.next.Before(now) {
		l.next = now
	}
	slot := l.next
	l.next = slot.Add(l.interval)
	l.mu.Unlock()

	delay := time.Until(slot)
	if delay <= 0 {
		return ctx.Err()
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(delay):
		return nil
	}
}

// penalize pushes the next allowed slot out, used when Notion answers 429.
func (l *limiter) penalize(d time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	until := time.Now().Add(d)
	if until.After(l.next) {
		l.next = until
	}
}
