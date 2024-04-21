package main

import (
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
)

var savedParamCount = 0

func saveParams(model *GPT2, filename string) {
	//if savedParamCount > 20 {
	//	os.Exit(1)
	//	return
	//}
	savedParamCount += 1
	if len(model.ParamsMemory) > 0 {
		name := fmt.Sprintf("checkpoint/go-params_memory-%s-%d", filename, savedParamCount)
		saveParams2(name, model.ParamsMemory)
	}
	if len(model.ActsMemory) > 0 {
		name := fmt.Sprintf("checkpoint/go-acts_memory-%s-%d", filename, savedParamCount)
		saveParams2(name, model.ActsMemory)
	}
	if len(model.GradsMemory) > 0 {
		name := fmt.Sprintf("checkpoint/go-grads_memory-%s-%d", filename, savedParamCount)
		saveParams2(name, model.GradsMemory)
	}

	if len(model.GradsActsMemory) > 0 {
		name := fmt.Sprintf("checkpoint/go-grads_acts_memory-%s-%d", filename, savedParamCount)
		saveParams2(name, model.GradsActsMemory)
	}
	//os.Exit(1)
}

func saveArr[t any](slice t) t {
	savedParamCount += 1
	//flat := flatten[float32](slice)
	name := fmt.Sprintf("go-slice-%d", savedParamCount)
	//fmt.Println(name, "len: ", len(flat))
	//saveParams2(name, flat)
	checkpointSave(name, slice)

	//sl := make([]float32, len(flat))
	sl := new(t)
	checkpointLoad(name, &sl)
	//if idx, eq := cmpArr(flat, sl); !eq {
	//	panic(fmt.Sprintf("not eq at: %v", idx))
	//}
	//fmt.Println("8474 from within saveArr: ", sl[8474])
	//fmt.Println("8452 from within saveArr: ", sl[8452])
	return *sl

}
func cmpArr[T comparable](a, b []T) (int, bool) {
	if len(a) != len(b) {
		return -1, false
	}
	for i := range a {
		if a[i] != b[i] {
			return i, false
		}
	}
	return 0, true
}

func loadArr(name string, n int) []float32 {
	goFile, err := os.Open(name)
	if err != nil {
		panic(err)
	}
	defer goFile.Close()
	m := make([]float32, n)
	if err := binary.Read(goFile, binary.LittleEndian, m); err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
		panic(err)
	}
	return m
}

func saveArrInt(slice any) {
	savedParamCount += 1
	flat := flatten[int32](slice)
	fmt.Println("len: ", len(flat))
	name := fmt.Sprintf("checkpoint/go-slice-%d", savedParamCount)
	saveParams2(name, flat)
}

func saveParams2[T any](name string, data T) {
	file, err := os.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.ModePerm)
	defer file.Close()
	if err != nil {
		panic(err)
	}
	if err := binary.Write(file, binary.LittleEndian, data); err != nil {
		panic(err)
	}
}

func SaveParamsMemory(params ParameterTensors) {
	savedParamCount += 1

	if len(params.WordTokEmbed.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-wte-%d", savedParamCount)
		arr := flatten[float32](params.WordTokEmbed)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.WordPosEmbed.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-wpe-%d", savedParamCount)
		arr := flatten[float32](params.WordPosEmbed)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.Layer1NormW.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-ln1w-%d", savedParamCount)
		arr := flatten[float32](params.Layer1NormW)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.Layer1NormB.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-ln1b-%d", savedParamCount)
		arr := flatten[float32](params.Layer1NormB)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.QueryKeyValW.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-qkvw-%d", savedParamCount)
		arr := flatten[float32](params.QueryKeyValW)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.QueryKeyValB.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-qkvb-%d", savedParamCount)
		arr := flatten[float32](params.QueryKeyValB)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.AttProjW.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-attprojw-%d", savedParamCount)
		arr := flatten[float32](params.AttProjW)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.AttProjB.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-attprojb-%d", savedParamCount)
		arr := flatten[float32](params.AttProjB)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.Layer2NormW.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-ln2w-%d", savedParamCount)
		arr := flatten[float32](params.Layer2NormW)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.Layer2NormB.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-ln2b-%d", savedParamCount)
		arr := flatten[float32](params.Layer2NormB)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.FeedFwdW.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-fcw-%d", savedParamCount)
		arr := flatten[float32](params.FeedFwdW)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.FeedFwdB.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-fcb-%d", savedParamCount)
		arr := flatten[float32](params.FeedFwdB)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.FeedFwdProjW.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-fcprojw-%d", savedParamCount)
		arr := flatten[float32](params.FeedFwdProjW)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.FeedFwdProjB.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-fcprojb-%d", savedParamCount)
		arr := flatten[float32](params.FeedFwdProjB)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.LayerFinNormW.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-lnfw-%d", savedParamCount)
		arr := flatten[float32](params.LayerFinNormW)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
	if len(params.LayerFinNormB.data) > 0 {
		name := fmt.Sprintf("checkpoint/go-params-lnfb-%d", savedParamCount)
		arr := flatten[float32](params.LayerFinNormB)
		fmt.Println(name, " : ", len(arr))
		saveParams2(name, arr)
	}
}

func byteToInt32Slice(sl []byte) []int32 {
	n := make([]int32, 0, len(sl))
	for _, elem := range sl {
		n = append(n, int32(elem))
	}
	return n
}

func assertSlicesEqual(flatSlice []float32, multiDimSlice interface{}, extra ...interface{}) {
	flattenedView := flatten[float32](multiDimSlice)

	// Compare lengths
	smaller := min(len(flatSlice), len(flattenedView))

	// Compare elements
	for i := 0; i < smaller; i++ {
		if flatSlice[i] != flattenedView[i] {
			panic(fmt.Sprintf("Slices have different elements at index: %d elem: %v != %v, extra:%v", i, flatSlice[i], flattenedView[i], extra))
		}
	}
}

func checkpointSave[t any](filename string, variable t) {
	// Create the file
	file, err := os.OpenFile("checkpoint/"+filename, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.ModePerm)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer file.Close() // Ensure file closure

	// Create a gob encoder
	encoder := gob.NewEncoder(file)

	// Encode and save the variable
	err = encoder.Encode(variable)
	if err != nil {
		fmt.Println(err)
		return
	}
}

func checkpointLoad(filename string, variable interface{}) {
	// Open the file
	filename = strings.ReplaceAll(filename, "checkpoint/", "")
	filename = "checkpoint/" + filename
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	// Create a gob decoder
	decoder := gob.NewDecoder(file)
	// Decode the variable from the file
	err = decoder.Decode(variable)
	if err != nil {
		panic(err)
	}
}

func checkTensor(aFlat, bFlat []float32, label string) bool {
	printUpto := 5
	failedPrint := 2 // print the first 2 failed elements
	ok := true
	fmt.Println(label)
	if len(aFlat) != len(bFlat) {
		fmt.Println("Slice lengths do not match")
		return false
	}
	for i := range aFlat {
		aFloat, bFloat := aFlat[i], bFlat[i]
		if diff, _, s := floatNearlyEqual(aFloat, bFloat); diff <= 1e-2 {
			if i < printUpto {
				fmt.Print("OK ")
			}
		} else {
			if i < printUpto || failedPrint > 0 {
				fmt.Printf("%s NOT OK %d %f %f\n", label, i, aFloat, bFloat)
				fmt.Println("not okay index:", i)
				fmt.Println(s)

				failedPrint -= 1
			}
			ok = false
		}
		if i < printUpto {
			fmt.Printf("%f %f\n", aFloat, bFloat)
		}
	}
	if ok {
		fmt.Println("TENSOR OK")
	} else {
		fmt.Println("TENSOR NOT OK")
	}
	return ok
}

// flatten takes a nested slice of any type and returns a flat slice of the same type.
func flatten[T any](input interface{}) []T {
	var flattened []T
	flattenRecursive(input, &flattened)
	return flattened
}

// flattenRecursive is a helper function that performs the recursive flattening.
func flattenRecursive[T any](input interface{}, result *[]T) {
	val := reflect.ValueOf(input)
	if val.Kind() == reflect.Slice {
		for i := 0; i < val.Len(); i++ {
			flattenRecursive(val.Index(i).Interface(), result)
		}
	} else {
		*result = append(*result, input.(T))
	}
}

func floatslice4dn(r []float32, a, b, c, d int) [][][][]float32 {
	s, _ := floatslice4d(r, a, b, c, d)
	return s
}

// Creates a 3D float slice
func floatslice3dn(r []float32, a, b, c int) [][][]float32 {
	s, _ := floatslice3d(r, a, b, c)
	return s
}

func floatslice2dn(r []float32, a, b int) [][]float32 {
	s, _ := floatslice2d(r, a, b)
	return s
}

func floatslice1dn(r []float32, a int) []float32 {
	s, _ := floatslice1d(r, a)
	return s
}

func floatslice3d(r []float32, depth, rows, cols int) ([][][]float32, int) {
	tensor := make([][][]float32, depth)
	var incr int
	for i := range tensor {
		tensor[i] = make([][]float32, rows)
		for j := range tensor[i] {
			tensor[i][j], incr = floatslice1d(r, cols)
			r = r[incr:]
		}
	}
	return tensor, depth * rows * cols
}

func floatslice4d(r []float32, dim1, dim2, dim3, dim4 int) ([][][][]float32, int) {
	volume := make([][][][]float32, dim1)
	var incr int
	for i := range volume {
		volume[i], incr = floatslice3d(r, dim2, dim3, dim4)
		r = r[incr:]
	}
	return volume, dim1 * dim2 * dim3 * dim4
}

func floatslice5d(r []float32, dim1, dim2, dim3, dim4, dim5 int) ([][][][][]float32, int) {
	volume := make([][][][][]float32, dim1)
	var incr int
	for i := range volume {
		volume[i], incr = floatslice4d(r, dim2, dim3, dim4, dim5)
		r = r[incr:]
	}
	return volume, dim1 * dim2 * dim3 * dim4 * dim5
}

// Creates a 1D float slice
func floatslice1d(m []float32, size int) ([]float32, int) {
	if size > len(m) {
		return make([]float32, size), size
	}
	return m[:size], size
}

// Creates a 2D float slice
func floatslice2d(r []float32, rows, cols int) ([][]float32, int) {
	matrix := make([][]float32, rows)
	var incr int
	for i := range matrix {
		matrix[i], incr = floatslice1d(r, cols)
		r = r[incr:]
	}
	return matrix, rows * cols
}

func floatToBinaryString(f float64) string {
	bits := math.Float64bits(f)
	return strconv.FormatUint(bits, 2)
}

func floatNearlyEqual(a, b float32) (difference float64, perc float64, s string) {
	epsilon := 1e-2
	// Calculate the absolute difference between the two values
	difference = math.Abs(float64(a) - float64(b))
	// Check if the difference is within the epsilon threshold
	perc = difference / float64(a)
	if difference < epsilon {
		return difference, perc, ""
	}
	s += fmt.Sprintf("  ∆: %10f\n", difference)
	s += fmt.Sprintf("  C: %f\n", a)
	s += fmt.Sprintf(" Go: %f\n", b)
	s += fmt.Sprintf("  C: %10f\n", a)
	s += fmt.Sprintf(" Go: %10f\n", b)
	s += fmt.Sprintf(" C%%: %10f%%\n", difference/float64(a))
	s += fmt.Sprintf("Go%%: %10f%%\n", difference/float64(b))
	s += fmt.Sprintf("fo%%: %10f%%\n", perc)
	s += fmt.Sprintf("%s\n%s\n", floatToBinaryString(float64(a)), floatToBinaryString(float64(b)))
	return difference, perc, s
}

func floatDiffString(a, b float32) (s string) {
	var difference, perc float64
	// Calculate the absolute difference between the two values
	difference = math.Abs(float64(a) - float64(b))
	// Check if the difference is within the epsilon threshold
	perc = difference / float64(a)
	s += fmt.Sprintf("  ∆: %10f\n", difference)
	s += fmt.Sprintf("  C: %f\n", a)
	s += fmt.Sprintf(" Go: %f\n", b)
	s += fmt.Sprintf("  C: %10f\n", a)
	s += fmt.Sprintf(" Go: %10f\n", b)
	s += fmt.Sprintf(" C%%: %10f%%\n", difference/float64(a))
	s += fmt.Sprintf("Go%%: %10f%%\n", difference/float64(b))
	s += fmt.Sprintf("fo%%: %10f%%\n", perc)
	s += fmt.Sprintf("%s\n%s\n", floatToBinaryString(float64(a)), floatToBinaryString(float64(b)))
	return
}

var foo int

func PrintModel(model *GPT2) {
	return
	if len(model.ParamsMemory) > 0 {
		fmt.Printf("%d model.ParamsMemory    : %10f %d\n", foo, Sum(model.ParamsMemory), len(model.ParamsMemory))
	}
	if len(model.ActsMemory) > 0 {
		fmt.Printf("%d model.ActsMemory      : %10f %d\n", foo, Sum(model.ActsMemory), len(model.ActsMemory))
	}
	if len(model.GradsMemory) > 0 {
		fmt.Printf("%d model.GradsMemory     : %10f %d\n", foo, Sum(model.GradsMemory), len(model.GradsMemory))
	}
	if len(model.GradsActsMemory) > 0 {
		fmt.Printf("%d model.GradsActsMemory : %10f %d\n", foo, Sum(model.GradsActsMemory), len(model.GradsActsMemory))
	}
	if len(model.MMemory) > 0 {
		fmt.Printf("%d model.MMemory         : %10f %d\n", foo, Sum(model.MMemory), len(model.MMemory))
	}
	if len(model.VMemory) > 0 {
		fmt.Printf("%d model.MMemory         : %10f %d\n", foo, Sum(model.VMemory), len(model.VMemory))
	}
}

func Sum(in []float32) (s float64) {
	for _, e := range in {
		s += float64(e)
	}
	return s
}

func PrintArr(name string, in []float32) {
	foo += 1
	if foo == 20 {
		print()
	}
	fmt.Printf("%s %d PrintArr %10f %d\n", name, foo, Sum(in), len(in))
}
