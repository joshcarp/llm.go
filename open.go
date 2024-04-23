//go:build !wasm

package llmgo

import (
	"io"
	"os"
)

func Open(name string) (io.ReadCloser, error) {
	return os.Open(name)
}
