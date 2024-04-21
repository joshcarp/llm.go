package main

import (
	"reflect"
	"testing"
)

func Test_flatten(t *testing.T) {
	type testCase[T any] struct {
		name string
		args any
		want []T
	}
	tests := []testCase[float32]{
		{
			args: [][]float32{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
			want: []float32{
				1, 2, 3, 4, 5, 6, 7, 8, 9,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := flatten[float32](tt.args); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("flatten() = %v, want %v", got, tt.want)
			}
		})
	}
}
