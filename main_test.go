package main

import (
	"encoding/binary"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestGPT(t *testing.T) {
	model, err := LoadGPT2Model("./data/gpt2_124M.bin")
	if err != nil {
		t.Error(err)
	}
	PrintModel(model)
	V := model.Config.VocabSize
	stateFile, err := os.Open("./data/gpt2_124M_debug_state.bin")
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
	x := make([]int32, B*T)
	y := make([]int32, B*T)
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
	mem := expectedGrads.init(model.Config.VocabSize, model.Config.Channels, model.Config.MaxSeqLen, model.Config.NumLayers)
	if err := binary.Read(stateFile, binary.LittleEndian, mem); err != nil {
		t.Error(err)
	}
	stateFile.Close()

	fmt.Printf("end of x: %d\n", y[B*T/2])
	fmt.Print(model)
	fmt.Printf("[State]\n")
	fmt.Printf("batch_size: %d\n", B)
	fmt.Printf("seq_len: %d\n", T)
	fmt.Printf("num_activations: %d\n", model.NumActivations)
	allok := true
	var losses []float32
	seen := map[*float32]struct{}{}
	for i := range model.GradsActsMemory {
		addr := &model.GradsActsMemory[i]
		if _, ok := seen[addr]; ok {
			panic("seen addr")
		}
		seen[addr] = struct{}{}
	}
	for i := range model.ParamsMemory {
		addr := &model.ParamsMemory[i]
		if _, ok := seen[addr]; ok {
			panic("seen addr")
		}
		seen[addr] = struct{}{}
	}
	for i := range model.ActsMemory {
		addr := &model.ActsMemory[i]
		if _, ok := seen[addr]; ok {
			panic("seen addr")
		}
		seen[addr] = struct{}{}
	}
	for i := range model.GradsMemory {
		addr := &model.GradsMemory[i]
		if _, ok := seen[addr]; ok {
			panic("seen addr")
		}
		seen[addr] = struct{}{}
	}

	for step := 0; step < 10; step++ {
		start := time.Now()
		model.forward(x, y, int(B), int(T))
		PrintModel(model)
		model.zeroGradient()
		PrintModel(model)
		model.backward()
		PrintModel(model)
		elapsed := time.Now().Sub(start)
		if step == 0 {
			logitsOk := true
			for i := 0; i < int(B)*int(T); i++ {
				if i < 3 {
					fmt.Printf("%f %f\n", expectedLogits[i], model.Acts.Logits.data[i])
				}
				if math.Abs(float64(expectedLogits[i]-model.Acts.Logits.data[i])) >= 1e-2 {
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
			// Compare the achieved loss
			if math.Abs(float64(model.MeanLoss-expectedLoss)) >= 1e-2 {
				fmt.Printf("LOSS MISMATCH: %f %f\n", model.MeanLoss, expectedLoss)
				allok = false // Update allok to false
			} else {
				fmt.Printf("LOSS OK: %f %f\n", model.MeanLoss, expectedLoss)
			}
			grads := model.Grads
			var ok bool
			ok = checkTensor(grads.WordTokEmbed.data, expectedGrads.WordTokEmbed.data, "dwte")
			allok = allok && ok
			ok = checkTensor(grads.WordPosEmbed.data, expectedGrads.WordPosEmbed.data, "dwpe")
			allok = allok && ok
			ok = checkTensor(grads.Layer1NormW.data, expectedGrads.Layer1NormW.data, "dln1w")
			allok = allok && ok
			ok = checkTensor(grads.Layer1NormB.data, expectedGrads.Layer1NormB.data, "dln1b")
			allok = allok && ok
			ok = checkTensor(grads.QueryKeyValW.data, expectedGrads.QueryKeyValW.data, "dqkvw")
			allok = allok && ok
			ok = checkTensor(grads.QueryKeyValB.data, expectedGrads.QueryKeyValB.data, "dqkvb")
			allok = allok && ok
			ok = checkTensor(grads.AttProjW.data, expectedGrads.AttProjW.data, "dattprojw")
			allok = allok && ok
			ok = checkTensor(grads.AttProjB.data, expectedGrads.AttProjB.data, "dattprojb")
			allok = allok && ok
			ok = checkTensor(grads.Layer2NormW.data, expectedGrads.Layer2NormW.data, "dln2w")
			allok = allok && ok
			ok = checkTensor(grads.Layer2NormB.data, expectedGrads.Layer2NormB.data, "dln2b")
			allok = allok && ok
			ok = checkTensor(grads.FeedFwdW.data, expectedGrads.FeedFwdW.data, "dfcw")
			allok = allok && ok
			ok = checkTensor(grads.FeedFwdB.data, expectedGrads.FeedFwdB.data, "dfcb")
			allok = allok && ok
			ok = checkTensor(grads.FeedFwdProjW.data, expectedGrads.FeedFwdProjW.data, "dfcprojw")
			allok = allok && ok
			ok = checkTensor(grads.FeedFwdProjB.data, expectedGrads.FeedFwdProjB.data, "dfcprojb")
			allok = allok && ok
			ok = checkTensor(grads.LayerFinNormW.data, expectedGrads.LayerFinNormW.data, "dlnfw")
			allok = allok && ok
			ok = checkTensor(grads.LayerFinNormB.data, expectedGrads.LayerFinNormB.data, "dlnfb")
			allok = allok && ok
		}
		model.update(1e-4, 0.999, 0.999, 1e-8, 0.01, step+1)
		fmt.Printf("step %d: loss %f (took %v ms)\n", step, model.MeanLoss, elapsed)
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
		if math.Abs(float64(losses[i]-expectedLosses[i])) >= 1e-2 {
			fmt.Printf("LOSS MISMATCH AT STEP %d: %f %f\n", i, losses[i], expectedLosses[i])
			allok = false
		} else {
			fmt.Printf("loss ok at step %d: %f %f\n", i, losses[i], expectedLosses[i])
		}
	}
	fmt.Printf("overall okay: %v\n", allok)
}

func TestWalk(t *testing.T) {
	filepath.Walk("checkpoint", func(path string, info fs.FileInfo, err error) error {
		fmt.Println(path)
		return nil
	})
}

func TestSmallGPT(t *testing.T) {
	//b := []byte{0xec, 0xfc, 0x50, 0x5d, 0x4f, 0x81, 0x37, 0x9a, 0x89, 0xeb, 0x97, 0x61, 0xb, 0xf8, 0xdd, 0xf3, 0xe5, 0x6c, 0x80, 0x9e, 0xe7, 0x4c, 0x96, 0x8d, 0xe9, 0xb4, 0x71, 0x71, 0xd6, 0xc3, 0x6b, 0x35, 0x58, 0xe5, 0x9b, 0xbe, 0xd4, 0xaa, 0xfc, 0x96, 0x14, 0x0, 0x79, 0xeb, 0x5a, 0x28, 0xac, 0xd1, 0x13, 0x70, 0x82, 0xf, 0x3, 0x8d, 0xbb, 0x4c, 0x51, 0x7e, 0xdb, 0x11, 0x24, 0x47, 0x4, 0x10, 0x9a, 0xa6, 0x69, 0x42, 0xd6, 0x6c, 0x33, 0x48, 0x5a, 0x7c, 0xe6, 0x73, 0xac, 0xde, 0x86, 0xd3, 0x52, 0x81, 0xf5, 0xa5, 0xe1, 0x8a, 0xaf, 0xac, 0x1e, 0xe9, 0xa9, 0x5e, 0xcb, 0xea, 0x12, 0x65, 0x8e, 0x6e, 0xa2, 0xef, 0x66, 0x63, 0x3e, 0x4b, 0x17, 0xab, 0x5f, 0xb, 0x2c, 0xdc, 0x5b, 0x40, 0x67, 0xe7, 0x90, 0xcd, 0x62, 0xde, 0x52, 0xb6, 0x6d, 0xea, 0xee, 0x4b, 0xe3, 0x70, 0x6c, 0x94, 0x68, 0xee, 0x34, 0x46, 0xcc, 0x0, 0xc1, 0x9, 0x77, 0xa4, 0x9f, 0x7, 0x78, 0xea, 0xc9, 0xd7, 0x32, 0x99, 0x39, 0xfc, 0xf7, 0x36, 0x29, 0x50, 0x90, 0xbd, 0x3c, 0xba, 0x1f, 0xdc, 0x4d, 0xaf, 0xf5, 0xc8, 0x55, 0x45, 0x20, 0xfd, 0x36, 0x97, 0xaa, 0xb, 0x58, 0xfe, 0xc8, 0x1f, 0x6d, 0xc1, 0x9f, 0xb0, 0x1a, 0xe6, 0x3d, 0x58, 0xbf, 0xa2, 0x78, 0x6, 0x27, 0x25, 0xd3, 0x3, 0xb, 0x94, 0x1, 0x19, 0x10, 0x5f, 0x7d, 0xe9, 0x98, 0x60, 0xc1, 0x3a, 0x10, 0xe1, 0x79, 0x50, 0x4b, 0xa0, 0xa6, 0x13, 0xc, 0x90, 0x28, 0x22, 0x95, 0xe1, 0x5f, 0x54, 0x82, 0x95, 0xce, 0x6d, 0x9d, 0xf2, 0x2, 0xa3, 0x87, 0xa7, 0x1a, 0x8d, 0xdf, 0xae, 0xe, 0x1a, 0x64, 0x1b, 0xb8, 0xa, 0xee, 0xa0, 0xa3, 0x95, 0x73, 0x38, 0xc5, 0xa5, 0x29, 0xb2, 0xf0, 0xce, 0xaf, 0x78, 0xc2, 0x3b, 0x2, 0xc6, 0x5a, 0x7f, 0x7a, 0x9b, 0x8c, 0x48, 0x63, 0x1c, 0xf1, 0x8f, 0x35, 0x4d, 0x75, 0x51, 0x66, 0xf9, 0x9, 0xb9, 0x43, 0x42, 0x79, 0x37, 0x67, 0xd6, 0xd, 0xdc, 0x6f, 0xcf, 0xc7, 0xd0, 0x7e, 0xbc, 0xca, 0xf7, 0xc0, 0x6c, 0xfc, 0xd1, 0x30, 0x43, 0x34, 0xe3, 0x16, 0x1, 0x1b, 0x90, 0xba, 0xcf, 0xce, 0x24, 0x21, 0x42, 0x71, 0xc0, 0x70, 0x16, 0x83, 0xc8, 0x78, 0xcf, 0xd0, 0x39, 0xdf, 0x7f, 0x81, 0xb2, 0x88, 0x27, 0x99, 0xc5, 0x65, 0x77, 0x3c, 0x29, 0x8c, 0xa3, 0xad, 0x82, 0x4a, 0x92, 0x72, 0xd8, 0x1d, 0xe2, 0x30, 0xc4, 0x4a, 0x43, 0xb, 0x52, 0x83, 0xb2, 0xe1, 0x94, 0x51, 0x8c, 0x8a, 0x4f, 0x3c, 0x72, 0xf5, 0x5, 0x84, 0x29, 0xed, 0xad, 0x73, 0x6e, 0x3b, 0xac, 0x65, 0x31, 0xdc, 0xfc, 0x5b, 0xf2, 0xca, 0xd4, 0xa7, 0x6a, 0xb1, 0x7, 0x96, 0xe, 0xab, 0xfe, 0xcc, 0x3c, 0xe7, 0x4, 0xa2, 0x1b, 0xcb, 0x82, 0xa6, 0x4a, 0x37, 0xef, 0x6f, 0x7c, 0x94, 0x79, 0xa5, 0x93, 0x24, 0xea, 0x4, 0x7b, 0xed, 0x28, 0x52, 0x2d, 0xcf, 0xc1, 0xa7, 0x89, 0xb3, 0x3b, 0x7b, 0x76, 0xde, 0xc3, 0xda, 0x3c, 0x73, 0x36, 0xa2, 0xc1, 0xdf, 0x37, 0xc5, 0xee, 0x48, 0xbd, 0x8b, 0xc8, 0x93, 0x12, 0xee, 0x74, 0x3a, 0xb6, 0x1e, 0xd3, 0x38, 0x83, 0x5, 0xe8, 0xfb, 0x46, 0x1b, 0xc8, 0xab, 0x81, 0x24, 0xaa, 0x97, 0xec, 0xbe, 0x17, 0x69, 0x39, 0x56, 0x45, 0xf9, 0x1d, 0xbc, 0xb6, 0xaf, 0xa2, 0x13, 0xe, 0x1f, 0x47, 0xd8, 0xc1, 0x20, 0x99, 0x12, 0xdc, 0x3d, 0xd2, 0x10, 0xec, 0xbb, 0xaa, 0x1b, 0x4c, 0x93, 0x2f, 0xd0, 0xa7, 0xdb, 0x1b, 0x5e, 0xdd, 0x1, 0x5d, 0x1, 0xe1, 0x79, 0x17, 0x93, 0x8b}
	B := 4
	T := 64
	model := GPT2{
		Config: GPT2Config{
			MaxSeqLen: 64,
			VocabSize: 50257,
			NumLayers: 2,
			NumHeads:  4,
			Channels:  200,
		},
	}

	model.ParamsMemory = model.Params.init(model.Config.VocabSize, model.Config.Channels, model.Config.MaxSeqLen, model.Config.NumLayers)
	model.NumParameters = len(model.ParamsMemory)
	var s = float32(0.0)
	for i := range model.ParamsMemory {
		model.ParamsMemory[i] = float32(0.001) * float32(i)
		s += model.ParamsMemory[i]
		//if (i % 100000) == 0 {
		//	fmt.Printf("hree: %f \n", s)
		//}
	}
	dataloader, err := NewDataLoader("data/tiny_shakespeare_val.bin", B, T)
	if err != nil {
		panic(err)
	}
	//dataloader2, err := NewDataLoader("data.txt", 2, 2)
	//if err != nil {
	//	panic(err)

	//}
	fmt.Printf("hree: %f ", s)

	for i := 0; i < 10; i++ {
		inp, tar, err := dataloader.NextBatch()
		if err != nil {
			panic(err)
		}
		//fmt.Println("numparams", model.NumParameters)
		PrintModel(&model)
		model.forward(inp, tar, B, T)
		PrintModel(&model)
		model.zeroGradient()
		PrintModel(&model)
		model.backward()
		PrintModel(&model)
		model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, i+1)
		PrintModel(&model)
	}

	//SaveParamsMemory(model.Params)

	// file, err := os.OpenFile("checkpoint/go-params", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	// if err != nil {
	// 	fmt.Println("Error opening file:", err)
	// 	return
	// }
	// Make sure to close the file when done
	// defer file.Close()

	// Write each float number from the array to the file
	// for _, floatNum := range model.ParamsMemory {
	// 	// Format the float number as a string and write it to the file
	// 	_, err := fmt.Fprintf(file, "%10f\n", floatNum)
	// 	if err != nil {
	// 		fmt.Println("Error writing to file:", err)
	// 		return
	// 	}
	// }

}
