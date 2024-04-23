//go:build wasm
// +build wasm

package main

import (
	"github.com/joshcarp/llmgo"
	"syscall/js"
	"time"
)

var model *llmgo.GPT2

func main() {
	done := make(chan struct{}, 0)
	js.Global().Set("chat", js.FuncOf(chat))
	//var err error
	//model, err = llmgo.LoadGPT2Model("https://huggingface.co/joshcarp/llm.go/resolve/main/gpt2_124M.bin?download=true", "https://huggingface.co/joshcarp/llm.go/resolve/main/gpt2_tokenizer.bin?download=true")
	//fmt.Println(err)
	<-done
}

func chat(this js.Value, args []js.Value) interface{} {
	controller := args[0]
	res := ""
	for _, str := range []string{"a", "b", "c", "d", "e", "f", "g"} {
		time.Sleep(time.Second)
		res += str
		jsStr := js.Global().Call("String", str).String()
		controller.Call("enqueue", jsStr)
	}
	return ""
}

//func chat(this js.Value, args []js.Value) interface{} {
//	res, err := model.Inference("a whole bunch of text that might work or might not work")
//	fmt.Println(err)
//	return res
//}

//
//func chat(this js.Value, args []js.Value, callback js.Func) interface{} {
//	for _, str := range []string{"a", "b", "c", "d", "e", "f", "g"} {
//		callback(str)
//	}
//	return nil // Or return something appropriate for your use case
//}
