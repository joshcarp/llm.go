package llmgo

import (
	"testing"
)

func TestSmallGPT(t *testing.T) {
	B := 4
	T := 64
	model := GPT2{
		Config: GPT2Config{
			MaxSeqLen: 64,
			V:         50257,
			L:         2,
			NH:        4,
			C:         200,
		},
	}
	model.Params.Init(model.Config.V, model.Config.C, model.Config.MaxSeqLen, model.Config.L)
	model.NumParameters = len(model.Params.Memory)
	var s float32
	for i := range model.Params.Memory {
		model.Params.Memory[i] = 0.001
		s += model.Params.Memory[i]
	}
	dataloader, err := NewDataLoader("./data/tiny_shakespeare_val.bin", B, T)
	if err != nil {
		panic(err)
	}
	for i := 0; i < 10; i++ {
		inp, tar, err := dataloader.NextBatch()
		if err != nil {
			panic(err)
		}
		model.Forward(inp, tar, B, T)
		model.ZeroGradient()
		model.Backward()
		model.Update(1e-4, 0.9, 0.999, 1e-8, 0.0, i+1)
	}
}
