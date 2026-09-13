//go:build unix

package embed

import (
	"os"
	"syscall"
)

// mapFile maps a file read-only and shared, so two processes that open the
// same weight blob occupy one set of physical pages.
func mapFile(f *os.File, size int) ([]byte, func() error, error) {
	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return syscall.Munmap(data) }, nil
}
