// Package notion is a small read-only client for the Notion REST API. It
// covers exactly what the mirror needs: discovery via search, data source
// queries, page metadata and block trees. Writes stay with the official
// Notion MCP server, so nothing here mutates a workspace.
package notion

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"time"
)

// DefaultVersion is the Notion API version this client speaks. 2025-09-03 is
// the first version with data sources split out of databases, which is the
// model the mirror stores.
const DefaultVersion = "2025-09-03"

// DefaultBaseURL is the public API root.
const DefaultBaseURL = "https://api.notion.com/v1"

// Options configures a Client.
type Options struct {
	Token   string
	BaseURL string
	Version string
	// RequestsPerSecond throttles outbound calls. Notion's documented average
	// is three per second; staying just under it avoids 429 storms during a
	// bootstrap crawl.
	RequestsPerSecond float64
	MaxRetries        int
	HTTPClient        *http.Client
	// Logf receives one line per retry or throttle event. May be nil.
	Logf func(format string, args ...any)
}

// Client talks to the Notion API.
type Client struct {
	opts    Options
	http    *http.Client
	limiter *limiter
}

// New builds a client. The token must be an internal integration secret with
// read access to the pages that should be mirrored.
func New(opts Options) (*Client, error) {
	if opts.Token == "" {
		return nil, errors.New("notion: empty API token")
	}
	if opts.BaseURL == "" {
		opts.BaseURL = DefaultBaseURL
	}
	if opts.Version == "" {
		opts.Version = DefaultVersion
	}
	if opts.RequestsPerSecond <= 0 {
		opts.RequestsPerSecond = 2.5
	}
	if opts.MaxRetries <= 0 {
		opts.MaxRetries = 5
	}
	hc := opts.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 60 * time.Second}
	}
	return &Client{opts: opts, http: hc, limiter: newLimiter(opts.RequestsPerSecond)}, nil
}

// APIError is a non-2xx response from Notion.
type APIError struct {
	Status  int
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *APIError) Error() string {
	return fmt.Sprintf("notion: HTTP %d %s: %s", e.Status, e.Code, e.Message)
}

// Retryable reports whether repeating the request could succeed.
func (e *APIError) Retryable() bool {
	return e.Status == http.StatusTooManyRequests || e.Status >= 500
}

func (c *Client) logf(format string, args ...any) {
	if c.opts.Logf != nil {
		c.opts.Logf(format, args...)
	}
}

// do performs one API call, retrying on 429 and 5xx with backoff.
func (c *Client) do(ctx context.Context, method, path string, body any) (json.RawMessage, error) {
	var payload []byte
	if body != nil {
		var err error
		payload, err = json.Marshal(body)
		if err != nil {
			return nil, err
		}
	}

	var lastErr error
	for attempt := 0; attempt <= c.opts.MaxRetries; attempt++ {
		if err := c.limiter.wait(ctx); err != nil {
			return nil, err
		}
		raw, err := c.attempt(ctx, method, path, payload)
		if err == nil {
			return raw, nil
		}
		lastErr = err

		var apiErr *APIError
		if !errors.As(err, &apiErr) || !apiErr.Retryable() || attempt == c.opts.MaxRetries {
			return nil, err
		}
		delay := backoff(attempt)
		if apiErr.Status == http.StatusTooManyRequests {
			c.limiter.penalize(delay)
		}
		c.logf("notion: %s %s failed (%v), retry %d in %s", method, path, err, attempt+1, delay)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
		}
	}
	return nil, lastErr
}

func (c *Client) attempt(ctx context.Context, method, path string, payload []byte) (json.RawMessage, error) {
	var rdr io.Reader
	if payload != nil {
		rdr = bytes.NewReader(payload)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.opts.BaseURL+path, rdr)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.opts.Token)
	req.Header.Set("Notion-Version", c.opts.Version)
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		apiErr := &APIError{Status: resp.StatusCode}
		_ = json.Unmarshal(data, apiErr)
		if apiErr.Message == "" {
			apiErr.Message = string(truncate(data, 300))
		}
		if ra := resp.Header.Get("Retry-After"); ra != "" {
			if secs, convErr := strconv.ParseFloat(ra, 64); convErr == nil {
				c.limiter.penalize(time.Duration(secs * float64(time.Second)))
			}
		}
		return nil, apiErr
	}
	return json.RawMessage(data), nil
}

func truncate(b []byte, n int) []byte {
	if len(b) <= n {
		return b
	}
	return b[:n]
}

// backoff returns an exponential delay capped at 30s.
func backoff(attempt int) time.Duration {
	d := time.Duration(math.Pow(2, float64(attempt))) * 500 * time.Millisecond
	if d > 30*time.Second {
		d = 30 * time.Second
	}
	return d
}
