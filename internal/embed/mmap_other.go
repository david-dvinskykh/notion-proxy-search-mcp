//go:build !unix

package embed

import (
	"io"
	"os"
)

// mapFile falls back to reading the whole blob where mmap is unavailable.
func mapFile(f *os.File, size int) ([]byte, func() error, error) {
	buf := make([]byte, size)
	if _, err := io.ReadFull(f, buf); err != nil {
		return nil, nil, err
	}
	return buf, func() error { return nil }, nil
}
