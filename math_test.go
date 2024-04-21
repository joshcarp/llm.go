package main

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

const delta = 1e-5

func Test_layernormForward(t *testing.T) {
	type args struct {
		inp    []float32
		weight []float32
		bias   []float32
		B      int
		T      int
		C      int
	}
	tests := []struct {
		name     string
		args     args
		wantOut  []float32
		wantMean []float32
		wantRstd []float32
	}{
		{
			name: "",
			args: args{
				inp:    []float32{0.2, 0.1, 0.3, 0.5, 0.1, 0.1},
				weight: []float32{1, 1, 1, 1, 1, 1},
				bias:   []float32{0, 0, 0, 0, 0, 0},
				B:      2,
				T:      1,
				C:      3,
			},
			wantOut:  []float32{0, -1.2238272, 1.2238274, 1.4140146, -0.70700747, -0.70700747},
			wantMean: []float32{0.2, 0.23333335},
			wantRstd: []float32{12.238273, 5.302555},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, mean, rstd := make([]float32, len(tt.args.inp)), make([]float32, tt.args.B*tt.args.T), make([]float32, tt.args.B*tt.args.T)
			layernormForward(out, mean, rstd, tt.args.inp, tt.args.weight, tt.args.bias, tt.args.B, tt.args.T, tt.args.C)
			require.InDeltaSlice(t, tt.wantOut, out, delta)
			require.InDeltaSlice(t, tt.wantMean, mean, delta)
			require.InDeltaSlice(t, tt.wantRstd, rstd, delta)
		})
	}
}

func Test_attentionForward(t *testing.T) {
	type args struct {
		inp []float32
		B   int
		T   int
		C   int
		NH  int
	}
	tests := []struct {
		name       string
		args       args
		wantOut    []float32
		wantPreatt []float32
		wantAtt    []float32
	}{
		{
			name: "Small Input Test",
			args: args{
				inp: []float32{1, 2, 3, 4, 5, 6},
				B:   1,
				T:   1,
				C:   2,
				NH:  1,
			},
			wantOut:    []float32{0.8, 1.2},
			wantPreatt: []float32{0.8},
			wantAtt:    []float32{1},
		},
		{
			name: "Larger Input Test",
			args: args{
				inp: []float32{ // (B, T, C3)
					/* B = 1 */
					/* T =  0 */
					/*qry*/ 1, 2, 3, // query compared against (4, 5, 6) but not (13, 14, 15) because it's in the future (t=1)
					/*key*/ 4, 5, 6,
					/*val*/ 7, 8, 9,
					/* T =  1 */
					/*qry*/ 10, 11, 12, // will be compared against (4, 5, 6) (t-1) and (13, 14, 15)
					/*key*/ 13, 14, 15,
					/*val*/ 16, 17, 18, // vals are updated to
				},
				B:  1,
				T:  2,
				C:  3,
				NH: 1,
			},
			wantOut: []float32{ // (B, T, C)
				/*      B = 0       */
				/*      T = 0       */
				/* C =  0    1    2 */
				/*  */ 7, 8, 9,
				/* T = 1 */
				/* C =  0    1    2 */
				/*  */ 16, 17, 18,
			},
			wantPreatt: []float32{ // (B, NH, T, T)
				/* B =  0    */
				/* NH = 0    */
				/*T =   1  2 */
				/*T=1*/ 18.475208, 0, // preatt: 18 -> 1, 0 -> 0
				/*T=2*/ 96.417496, 267.89053, // 96 -> 9, 267 -> 1
			},
			wantAtt: []float32{ // (B, NH, T, T)
				/* B = 0     */
				/* NH = 0    */
				/*T =   1  2 */
				/*T=1*/ 1, 0,
				/*T=2*/ 0, 1,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, preatt, att := make([]float32, len(tt.wantOut)), make([]float32, len(tt.wantPreatt)), make([]float32, len(tt.wantAtt))
			attentionForward(out, preatt, att, tt.args.inp, tt.args.B, tt.args.T, tt.args.C, tt.args.NH)
			assert.InDeltaSlice(t, tt.wantOut, out, 1e-4, fmt.Sprintf("want: %v got: %v", tt.wantOut, out))
			assert.InDeltaSlice(t, tt.wantPreatt, preatt, 1e-4, fmt.Sprintf("want: %v got: %v", tt.wantPreatt, preatt))
			assert.InDeltaSlice(t, tt.wantAtt, att, 1e-4, fmt.Sprintf("want: %v got: %v", tt.wantAtt, att))
		})
	}
}

func Test_matmulForward(t *testing.T) {
	type args struct {
		inp    []float32
		weight []float32
		bias   []float32
		B      int
		T      int
		C      int
		OC     int
	}
	tests := []struct {
		name    string
		args    args
		wantOut []float32
	}{
		{
			name: "simple",
			args: args{
				weight: []float32{ // OC (3) * C(2)
					1, 2,
					3, 4,
					5, 6,
				},
				inp: []float32{ // B(1) * T(1) * T(1) * C(2)
					1,
					2,
				},
				bias: []float32{1, 2, 3}, // OC
				// WEIGHT * INP + BIAS
				B:  1,
				T:  1,
				C:  2,
				OC: 3,
			},
			wantOut: []float32{
				6,
				13,
				20,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := make([]float32, tt.args.OC)
			matmulForward(out, tt.args.inp, tt.args.weight, tt.args.bias, tt.args.B, tt.args.T, tt.args.C, tt.args.OC)
			assert.Equal(t, tt.wantOut, out)
		})
	}
}

func Test_residualBackward(t *testing.T) {
	type args struct {
		dinp1 []float32
		dinp2 []float32
		dout  []float32
		N     int
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			residualBackward(tt.args.dinp1, tt.args.dinp2, tt.args.dout, tt.args.N)
		})
	}
}

func Test_residualForward(t *testing.T) {
	type args struct {
		out  []float32
		inp1 []float32
		inp2 []float32
		N    int
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			residualForward(tt.args.out, tt.args.inp1, tt.args.inp2, tt.args.N)
		})
	}
}

func Test_geluBackward(t *testing.T) {
	type args struct {
		dinp []float32
		inp  []float32
		dout []float32
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			geluBackward(tt.args.dinp, tt.args.inp, tt.args.dout, 0)
		})
	}
}

func Test_geluForward(t *testing.T) {
	type args struct {
		out []float32
		inp []float32
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			geluForward(tt.args.out, tt.args.inp, 0)
		})
	}
}

func Test_softmaxForward(t *testing.T) {
	type args struct {
		probs  []float32
		logits []float32
		B      int
		T      int
		V      int
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			softmaxForward(tt.args.probs, tt.args.logits, tt.args.B, tt.args.T, tt.args.V)
		})
	}
}

func Test_crossentropySoftmaxBackward(t *testing.T) {
	type args struct {
		dlogits []float32
		dlosses []float32
		probs   []float32
		targets []int32
		B       int
		T       int
		V       int
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			crossentropySoftmaxBackward(tt.args.dlogits, tt.args.dlosses, tt.args.probs, tt.args.targets, tt.args.B, tt.args.T, tt.args.V)
		})
	}
}

func Test_crossEntropyForward(t *testing.T) {
	type args struct {
		losses  []float32
		probs   []float32
		targets []int32
		B       int
		T       int
		V       int
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			crossEntropyForward(tt.args.losses, tt.args.probs, tt.args.targets, tt.args.B, tt.args.T, tt.args.V)
		})
	}
}

func Test_matmulBackward(t *testing.T) {
	type args struct {
		dinp    []float32
		dweight []float32
		dbias   []float32
		dout    []float32
		inp     []float32
		weight  []float32
		B       int
		T       int
		C       int
		OC      int
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			matmulBackward(tt.args.dinp, tt.args.dweight, tt.args.dbias, tt.args.dout, tt.args.inp, tt.args.weight, tt.args.B, tt.args.T, tt.args.C, tt.args.OC)
		})
	}
}

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

func Test_layernormBackward(t *testing.T) {
	type args struct {
		dinp    []float32
		dweight []float32
		dbias   []float32
		dout    []float32
		inp     []float32
		weight  []float32
		mean    []float32
		rstd    []float32
		B       int
		T       int
		C       int
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layernormBackward(tt.args.dinp, tt.args.dweight, tt.args.dbias, tt.args.dout, tt.args.inp, tt.args.weight, tt.args.mean, tt.args.rstd, tt.args.B, tt.args.T, tt.args.C)
		})
	}
}

func Test_attentionBackward(t *testing.T) {
	type args struct {
		dinp    []float32
		dpreatt []float32
		datt    []float32
		dout    []float32
		inp     []float32
		att     []float32
		B       int
		T       int
		C       int
		NH      int
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			attentionBackward(tt.args.dinp, tt.args.dpreatt, tt.args.datt, tt.args.dout, tt.args.inp, tt.args.att, tt.args.B, tt.args.T, tt.args.C, tt.args.NH)
		})
	}
}

func Test_encoderBackward(t *testing.T) {
	type args struct {
		out  []float32
		inp  []int32
		dwte []float32
		dwpe []float32
		dout []float32
		B    int
		T    int
		C    int
	}
	tests := []struct {
		name     string
		args     args
		wantdwte []float32
		wantdwpe []float32
	}{
		{
			name: "",
			args: args{
				inp:  []int32{1}, //  [0 -> wte (3, 4), wpe(6, 7) (position 0)]
				dwte: []float32{1, 2, 3, 4},
				dwpe: []float32{6, 7, 8, 9},
				dout: []float32{1, 2, 3, 4}, // contains the diff that will be applied to wte and
				B:    1,                     // Batch size
				T:    1,                     // Sequence Len
				C:    2,                     // Dimensions
			},
			wantdwte: []float32{1, 2, 4, 6}, // 3, 4 (wte[inp[0]]) + dout[0]
			wantdwpe: []float32{7, 9, 8, 9},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoderBackward(tt.args.dwte, tt.args.dwpe, tt.args.dout, tt.args.inp, tt.args.B, tt.args.T, tt.args.C)
			assert.Equal(t, tt.wantdwpe, tt.args.dwpe)
			assert.Equal(t, tt.wantdwte, tt.args.dwte)
		})
	}
}

func Test_encoderForward(t *testing.T) {
	type args struct {
		out []float32
		inp []int32
		wte []float32
		wpe []float32
		B   int
		T   int
		C   int
	}
	tests := []struct {
		name    string
		args    args
		wantOut []float32
	}{
		{
			name: "",
			args: args{
				inp: []int32{1, 0}, // [1 -> wte (2, 3), wpe(4, 5)] [0 -> wte (0, 1), wpe(6, 7)]
				wte: []float32{0, 1, 2, 3},
				wpe: []float32{4, 5, 6, 7},
				B:   1, // Batch size
				T:   1, // Sequence Len
				C:   2, // Dimensions
			},
			wantOut: []float32{6, 8},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := make([]float32, len(tt.args.inp))
			encoderForward(out, tt.args.inp, tt.args.wte, tt.args.wpe, tt.args.B, tt.args.T, tt.args.C)
			assert.Equal(t, tt.wantOut, out)
		})
	}
}

func Test_geluForward1(t *testing.T) {
	type args struct {
		out []float32
		inp []float32
	}
	tests := []struct {
		name string
		args args
	}{
		{},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			geluForward(tt.args.out, tt.args.inp, 0)
		})
	}
}
