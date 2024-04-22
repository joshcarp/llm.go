package main

import (
	"fmt"
)

func checkTensor(tensorA, tensorB []float32, label string) bool {
	printUpto, failedPrint := 5, 2
	ok := true
	fmt.Println(label)
	if len(tensorA) != len(tensorB) {
		fmt.Println("Slice lengths do not match")
		return false
	}
	for i := range tensorA {
		a, b := tensorA[i], tensorB[i]
		if diff := Abs(a - b); diff <= 1e-2 {
			if i < printUpto {
				fmt.Print("OK ")
			}
		} else {
			if i < printUpto || failedPrint > 0 {
				fmt.Printf("%s NOT OK %d %f %f\n", label, i, a, b)
				fmt.Println("NOT OK AT INDEX:", i)
				fmt.Printf("  ∆: %10f\n", diff)
				fmt.Printf("  C: %f\n", a)
				fmt.Printf(" Go: %f\n", b)
				failedPrint -= 1
			}
			ok = false
		}
		if i < printUpto {
			fmt.Printf("%f %f\n", a, b)
		}
	}
	if ok {
		fmt.Println("TENSOR OK")
	} else {
		fmt.Println("TENSOR NOT OK")
	}
	return ok
}

func checkParameters(got, want ParameterTensors) bool {
	ok, allok := true, true
	ok = checkTensor(got.WordTokEmbed.data, want.WordTokEmbed.data, "dwte")
	allok = allok && ok
	ok = checkTensor(got.WordPosEmbed.data, want.WordPosEmbed.data, "dwpe")
	allok = allok && ok
	ok = checkTensor(got.LayerNorm1W.data, want.LayerNorm1W.data, "dln1w")
	allok = allok && ok
	ok = checkTensor(got.LayerNorm1B.data, want.LayerNorm1B.data, "dln1b")
	allok = allok && ok
	ok = checkTensor(got.QueryKeyValW.data, want.QueryKeyValW.data, "dqkvw")
	allok = allok && ok
	ok = checkTensor(got.QueryKeyValB.data, want.QueryKeyValB.data, "dqkvb")
	allok = allok && ok
	ok = checkTensor(got.AttProjW.data, want.AttProjW.data, "dattprojw")
	allok = allok && ok
	ok = checkTensor(got.AttProjB.data, want.AttProjB.data, "dattprojb")
	allok = allok && ok
	ok = checkTensor(got.Layer2NormW.data, want.Layer2NormW.data, "dln2w")
	allok = allok && ok
	ok = checkTensor(got.Layer2NormB.data, want.Layer2NormB.data, "dln2b")
	allok = allok && ok
	ok = checkTensor(got.FeedFwdW.data, want.FeedFwdW.data, "dfcw")
	allok = allok && ok
	ok = checkTensor(got.FeedFwdB.data, want.FeedFwdB.data, "dfcb")
	allok = allok && ok
	ok = checkTensor(got.FeedFwdProjW.data, want.FeedFwdProjW.data, "dfcprojw")
	allok = allok && ok
	ok = checkTensor(got.FeedFwdProjB.data, want.FeedFwdProjB.data, "dfcprojb")
	allok = allok && ok
	ok = checkTensor(got.LayerFinNormW.data, want.LayerFinNormW.data, "dlnfw")
	allok = allok && ok
	ok = checkTensor(got.LayerFinNormB.data, want.LayerFinNormB.data, "dlnfb")
	return ok && allok
}
