package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_sampleMult(t *testing.T) {
	type args struct {
		probabilities []float32
		coin          float32
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equalf(t, tt.want, sampleMult(tt.args.probabilities, tt.args.coin), "sampleMult(%v, %v)", tt.args.probabilities, tt.args.coin)
		})
	}
}
