//go:build wasm
// +build wasm

package main

import (
	"fmt"
	"github.com/joshcarp/llmgo"
	"syscall/js"
)

var model *llmgo.GPT2

func main() {
	done := make(chan struct{}, 0)
	js.Global().Set("chat", js.FuncOf(chat))
	var err error
	model, err = llmgo.LoadGPT2Model("./data/gpt2_124M.bin", "./data/gpt2_tokenizer.bin")
	fmt.Println(err)
	<-done
}

func chat(this js.Value, args []js.Value) interface{} {
	res, err := model.Inference("a whole bunch of text that might work or might not work")
	fmt.Println(err)
	return res
}
