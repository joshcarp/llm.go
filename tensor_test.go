package main

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_tensor_index(t1 *testing.T) {
	type args struct {
		idx []int
	}
	type testCase struct {
		name string
		t    tensor[float32]
		args args
		want tensor[float32]
	}
	tests := []testCase{
		{
			name: "",
			t: tensor[float32]{
				data: []float32{1, 2, 3, 4},
				dims: []int{2, 2},
			},
			args: args{
				idx: []int{1},
			},
			want: tensor[float32]{
				data: []float32{3, 4},
				dims: []int{2},
			},
		},
		{
			name: "",
			t: tensor[float32]{
				data: []float32{1, 2, 3, 4},
				dims: []int{2, 2},
			},
			args: args{
				idx: []int{0},
			},
			want: tensor[float32]{
				data: []float32{1, 2},
				dims: []int{2},
			},
		},
	}
	for _, tt := range tests {
		t1.Run(tt.name, func(t1 *testing.T) {
			got := tt.t.index(tt.args.idx...)
			assert.Equalf(t1, tt.want, got, "index(%v)", tt.args.idx)
		})
	}
}

func TestParameterTensors_init(t *testing.T) {
	type args struct {
		vocabSize int
		channels  int
		maxSeqLen int
		numLayers int
	}
	tests := []struct {
		name string
		args args
		want ParameterTensors
	}{
		{
			name: "",
			args: args{
				vocabSize: 1,
				channels:  1,
				maxSeqLen: 1,
				numLayers: 1,
			},
			want: ParameterTensors{
				WordTokEmbed:  tensor[float32]{data: []float32{0}, dims: []int{1, 1}},
				WordPosEmbed:  tensor[float32]{data: []float32{1}, dims: []int{1, 1}},
				Layer1NormW:   tensor[float32]{data: []float32{2}, dims: []int{1, 1}},
				Layer1NormB:   tensor[float32]{data: []float32{3}, dims: []int{1, 1}},
				QueryKeyValW:  tensor[float32]{data: []float32{4, 5, 6}, dims: []int{1, 3, 1}},
				QueryKeyValB:  tensor[float32]{data: []float32{7, 8, 9}, dims: []int{1, 3}},
				AttProjW:      tensor[float32]{data: []float32{10}, dims: []int{1, 1, 1}},
				AttProjB:      tensor[float32]{data: []float32{11}, dims: []int{1, 1}},
				Layer2NormW:   tensor[float32]{data: []float32{12}, dims: []int{1, 1}},
				Layer2NormB:   tensor[float32]{data: []float32{13}, dims: []int{1, 1}},
				FeedFwdW:      tensor[float32]{data: []float32{14, 15, 16, 17}, dims: []int{1, 4, 1}},
				FeedFwdB:      tensor[float32]{data: []float32{18, 19, 20, 21}, dims: []int{1, 4}},
				FeedFwdProjW:  tensor[float32]{data: []float32{22, 23, 24, 25}, dims: []int{1, 1, 4}},
				FeedFwdProjB:  tensor[float32]{data: []float32{26}, dims: []int{1, 1}},
				LayerFinNormW: tensor[float32]{data: []float32{27}, dims: []int{1}},
				LayerFinNormB: tensor[float32]{data: []float32{28}, dims: []int{1}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := ParameterTensors{}
			got := tensor.init(tt.args.vocabSize, tt.args.channels, tt.args.maxSeqLen, tt.args.numLayers)
			for i := range got {
				got[i] = float32(i)
			}
			assert.Equal(t, tt.want, tensor)
		})
	}
}

func Test_newTensor(t *testing.T) {
	type args[T any] struct {
		data []T
		dims []int
	}
	type testCase[T any] struct {
		name  string
		args  args[T]
		want  tensor[T]
		want1 int
	}
	tests := []testCase[float32]{
		{
			name: "",
			args: args[float32]{
				data: []float32{
					1, 2, 3, 4, 5,
				},
				dims: []int{
					1, 2,
				},
			},
			want1: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1 := newTensor(tt.args.data, tt.args.dims...)
			assert.Equalf(t, tt.want, got, "newTensor(%v, %v)", tt.args.data, tt.args.dims)
			assert.Equalf(t, tt.want1, got1, "newTensor(%v, %v)", tt.args.data, tt.args.dims)
		})
	}
}
