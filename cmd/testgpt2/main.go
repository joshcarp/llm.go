package main

import (
	"encoding/binary"
	"fmt"
	"github.com/joshcarp/llmgo"
	"log"
	"time"
)

func main() {
	model, err := llmgo.LoadGPT2Model("./gpt2_124M.bin", "./gpt2_tokenizer.bin")
	if err != nil {
		log.Fatal(err)
	}
	V := model.Config.V
	stateFile, err := llmgo.Open("./gpt2_124M_debug_state.bin")
	if err != nil {
		log.Fatal(err)
	}
	stateHeader := make([]int32, 256)
	if err := binary.Read(stateFile, binary.LittleEndian, stateHeader); err != nil {
		log.Fatal(err)
	}
	if stateHeader[0] != 20240327 {
		log.Fatal("Bad magic state file")
	}
	if stateHeader[1] != 1 {
		log.Fatal("Bad version in state file")
	}
	B, T := stateHeader[2], stateHeader[3] // batch size, time / sequence length (e.g. 64, up to maxT)
	x, y := make([]int32, B*T), make([]int32, B*T)
	expectedLogits := make([]float32, B*T*int32(V))
	var expectedLoss float32
	if err := binary.Read(stateFile, binary.LittleEndian, x); err != nil {
		log.Fatal(err)
	}
	if err := binary.Read(stateFile, binary.LittleEndian, y); err != nil {
		log.Fatal(err)
	}
	if err := binary.Read(stateFile, binary.LittleEndian, expectedLogits); err != nil {
		log.Fatal(err)
	}
	if err := binary.Read(stateFile, binary.LittleEndian, &expectedLoss); err != nil {
		log.Fatal(err)
	}
	var expectedGrads llmgo.ParameterTensors
	expectedGrads.Init(model.Config.V, model.Config.C, model.Config.MaxSeqLen, model.Config.L)
	if err := binary.Read(stateFile, binary.LittleEndian, expectedGrads.Memory); err != nil {
		log.Fatal(err)
	}
	stateFile.Close()
	fmt.Print(model)
	fmt.Printf("[State]\n")
	fmt.Printf("batch_size: %d\n", B)
	fmt.Printf("seq_len: %d\n", T)
	fmt.Printf("num_activations: %d\n", len(model.Acts.Memory))
	allok := true
	var losses []float32
	for step := 0; step < 10; step++ {
		start := time.Now()
		model.Forward(x, y, int(B), int(T))
		model.ZeroGradient()
		if err := model.Backward(); err != nil {
			log.Fatal(err)
		}
		elapsed := time.Now().Sub(start)
		if step == 0 {
			logitsOk := true
			for i := 0; i < int(B)*int(T); i++ {
				if i < 3 {
					fmt.Printf("%f %f\n", expectedLogits[i], model.Acts.Logits.Data()[i])
				}
				if llmgo.Abs(expectedLogits[i]-model.Acts.Logits.Data()[i]) >= 1e-2 {
					fmt.Printf("step: %d MISMATCH AT INDEX %d: %f %f\n", step, i, expectedLogits[i], model.Acts.Logits.Data()[i])
					logitsOk = false
					break
				}
			}
			if !logitsOk {
				fmt.Print("NOT ")
			}
			fmt.Println("OK (LOGITS)")
			allok = allok && logitsOk
			if llmgo.Abs(model.MeanLoss-expectedLoss) >= 1e-2 {
				fmt.Printf("LOSS MISMATCH: %f %f\n", model.MeanLoss, expectedLoss)
				allok = false
			} else {
				fmt.Printf("LOSS OK: %f %f\n", model.MeanLoss, expectedLoss)
			}
			allok = llmgo.CheckParameters(model.Grads, expectedGrads) && allok
		}
		model.Update(1e-4, 0.9, 0.999, 1e-8, 0.01, step+1)
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
		if llmgo.Abs(losses[i]-expectedLosses[i]) >= 1e-2 {
			fmt.Printf("LOSS MISMATCH AT STEP %d: %f %f\n", i, losses[i], expectedLosses[i])
			allok = false
		} else {
			fmt.Printf("loss ok at step %d: %f %f\n", i, losses[i], expectedLosses[i])
		}
	}
	fmt.Printf("overall okay: %v\n", allok)
}
