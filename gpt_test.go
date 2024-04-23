package llmgo

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestLoadGPT2Model(t *testing.T) {
	tests := []struct {
		name      string
		maxSeqLen int
		v         int
		l         int
		nh        int
		c         int
		vocab     []string
		input     string
		output    string
	}{
		{
			name:      "",
			maxSeqLen: 3,
			v:         3,
			l:         2,
			nh:        1,
			c:         1,
			vocab:     []string{"a", "b", "c"},
			input:     "abcd",
			output:    "acc",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			B, T := 1, 2
			model := newGPT2(tt.maxSeqLen, tt.v, tt.l, tt.nh, tt.c, tt.vocab)
			tokens, err := model.Tokenizer.Encode(tt.input)
			validation, err := newDataLoaderFromInts(tokens, B, T)
			assert.NoError(t, err)
			train, err := newDataLoaderFromInts(tokens, B, T)
			assert.NoError(t, err)
			err = model.Train(validation, train, B, T)
			assert.NoError(t, err)
			output, err := model.Inference(tt.input, 1, 2)
			assert.NoError(t, err)
			println(output)
		})
	}
}
