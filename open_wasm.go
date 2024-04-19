//go:build wasm
// +build wasm

package llmgo

import (
	"bytes"
	"errors"
	"io"
	"syscall/js"
)

func Open(name string) (io.ReadCloser, error) {
	// Create a channel to receive the result
	resultChan := make(chan []byte, 1)

	// Make an HTTP request to retrieve the file
	resp := js.Global().Call("fetch", name)

	// Define a JavaScript function to handle the response asynchronously
	resolveFunc := js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		// Get the response object
		resp := args[0]

		// Asynchronously read the response body as array buffer
		resp.Call("arrayBuffer").Call("then", js.FuncOf(func(v js.Value, x []js.Value) any {
			data := js.Global().Get("Uint8Array").New(x[0])
			dst := make([]byte, data.Get("length").Int())
			js.CopyBytesToGo(dst, data)
			resultChan <- dst
			return nil
		}))

		return nil
	})

	// Attach the resolve function to the promise's then method
	resp.Call("then", resolveFunc)

	// Wait for the result or timeout
	select {
	case text := <-resultChan:
		// Create a Reader from the string contents
		reader := bytes.NewReader(text)
		return io.NopCloser(reader), nil
		// For simplicity, we don't handle timeout here
	}
	// Return an error if there's no result
	return nil, errors.New("timeout waiting for response")
}
