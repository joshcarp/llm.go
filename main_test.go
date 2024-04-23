package llmgo

import (
	"encoding/binary"
	"fmt"
	"testing"
	"time"
)

func TestGPT(t *testing.T) {
	model, err := LoadGPT2Model("./data/gpt2_124M.bin", "./data/gpt2_tokenizer.bin")
	if err != nil {
		t.Error(err)
	}
	V := model.Config.V
	stateFile, err := Open("./data/gpt2_124M_debug_state.bin")
	if err != nil {
		t.Error(err)
	}
	stateHeader := make([]int32, 256)
	if err := binary.Read(stateFile, binary.LittleEndian, stateHeader); err != nil {
		t.Error(err)
	}
	if stateHeader[0] != 20240327 {
		t.Error("Bad magic state file")
	}
	if stateHeader[1] != 1 {
		t.Error("Bad version in state file")
	}
	B, T := stateHeader[2], stateHeader[3] // batch size, time / sequence length (e.g. 64, up to maxT)
	x, y := make([]int32, B*T), make([]int32, B*T)
	expectedLogits := make([]float32, B*T*int32(V))
	var expectedLoss float32
	if err := binary.Read(stateFile, binary.LittleEndian, x); err != nil {
		t.Error(err)
	}
	if err := binary.Read(stateFile, binary.LittleEndian, y); err != nil {
		t.Error(err)
	}
	if err := binary.Read(stateFile, binary.LittleEndian, expectedLogits); err != nil {
		t.Error(err)
	}
	if err := binary.Read(stateFile, binary.LittleEndian, &expectedLoss); err != nil {
		t.Error(err)
	}
	var expectedGrads ParameterTensors
	expectedGrads.init(model.Config.V, model.Config.C, model.Config.MaxSeqLen, model.Config.L)
	if err := binary.Read(stateFile, binary.LittleEndian, expectedGrads.Memory); err != nil {
		t.Error(err)
	}
	stateFile.Close()
	fmt.Print(model)
	fmt.Printf("[State]\n")
	fmt.Printf("batch_size: %d\n", B)
	fmt.Printf("seq_len: %d\n", T)
	fmt.Printf("num_activations: %d\n", model.NumActivations)
	allok := true
	var losses []float32
	for step := 0; step < 10; step++ {
		start := time.Now()
		model.forward(x, y, int(B), int(T))
		model.zeroGradient()
		if err := model.backward(); err != nil {
			t.Error(err)
		}
		elapsed := time.Now().Sub(start)
		if step == 0 {
			logitsOk := true
			for i := 0; i < int(B)*int(T); i++ {
				if i < 3 {
					fmt.Printf("%f %f\n", expectedLogits[i], model.Acts.Logits.data[i])
				}
				if Abs(expectedLogits[i]-model.Acts.Logits.data[i]) >= 1e-2 {
					fmt.Printf("step: %d MISMATCH AT INDEX %d: %f %f\n", step, i, expectedLogits[i], model.Acts.Logits.data[i])
					logitsOk = false
					break
				}
			}
			if !logitsOk {
				fmt.Print("NOT ")
			}
			fmt.Println("OK (LOGITS)")
			allok = allok && logitsOk
			if Abs(model.MeanLoss-expectedLoss) >= 1e-2 {
				fmt.Printf("LOSS MISMATCH: %f %f\n", model.MeanLoss, expectedLoss)
				allok = false
			} else {
				fmt.Printf("LOSS OK: %f %f\n", model.MeanLoss, expectedLoss)
			}
			allok = checkParameters(model.Grads, expectedGrads) && allok
		}
		model.update(1e-4, 0.9, 0.999, 1e-8, 0.01, step+1)
		fmt.Printf("step %d: loss %f (took %v)\n", step, model.MeanLoss, elapsed)
		losses = append(losses, model.MeanLoss)
	}
	// expected losses are as follows, from Python
	expectedLosses := []float32{
		5.270007133483887,
		4.059706687927246,
		3.3751230239868164,
		2.8007826805114746,
		2.315382242202759,
		1.8490285873413086,
		1.3946564197540283,
		0.9991465210914612,
		0.6240804195404053,
		0.37651097774505615,
	}
	for i := range expectedLosses {
		if Abs(losses[i]-expectedLosses[i]) >= 1e-2 {
			fmt.Printf("LOSS MISMATCH AT STEP %d: %f %f\n", i, losses[i], expectedLosses[i])
			allok = false
		} else {
			fmt.Printf("loss ok at step %d: %f %f\n", i, losses[i], expectedLosses[i])
		}
	}
	fmt.Printf("overall okay: %v\n", allok)
}

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
	model.Params.init(model.Config.V, model.Config.C, model.Config.MaxSeqLen, model.Config.L)
	model.NumParameters = len(model.Params.Memory)
	var s float32
	for i := range model.Params.Memory {
		model.Params.Memory[i] = 0.001
		s += model.Params.Memory[i]
	}
	dataloader, err := NewDataLoader("data/tiny_shakespeare_val.bin", B, T)
	if err != nil {
		panic(err)
	}
	for i := 0; i < 10; i++ {
		inp, tar, err := dataloader.NextBatch()
		if err != nil {
			panic(err)
		}
		model.forward(inp, tar, B, T)
		model.zeroGradient()
		model.backward()
		model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, i+1)
	}
}
