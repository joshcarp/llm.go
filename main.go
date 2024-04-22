package main

import (
	"fmt"
	"log"
)

func main() {
	model, err := LoadGPT2Model("./data/gpt2_124M.bin", "./data/gpt2_tokenizer.bin")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(model)
	B, T := 4, 64
	trainDataloader, err := NewDataLoader("data/tiny_shakespeare_train.bin", B, T)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("train dataset num_batches: %d\n", trainDataloader.numBatches)
	valDataloader, err := NewDataLoader("data/tiny_shakespeare_val.bin", B, T)
	if err != nil {
		log.Fatal(err)
	}
	if err := model.train(valDataloader, trainDataloader, B, T); err != nil {
		log.Fatal(err)
	}
}
