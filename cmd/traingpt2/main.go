package main

import (
	"fmt"
	"github.com/joshcarp/llmgo"
	"log"
)

func main() {
	model, err := llmgo.LoadGPT2Model("./gpt2_124M.bin", "./gpt2_tokenizer.bin")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(model)
	B, T := 4, 64
	trainDataloader, err := llmgo.NewDataLoader("./data/tiny_shakespeare_train.bin", B, T)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("train dataset num_batches: %d\n", trainDataloader.NumBatches)
	valDataloader, err := llmgo.NewDataLoader("./data/tiny_shakespeare_val.bin", B, T)
	if err != nil {
		log.Fatal(err)
	}
	if err := model.Train(valDataloader, trainDataloader, B, T); err != nil {
		log.Fatal(err)
	}
}
