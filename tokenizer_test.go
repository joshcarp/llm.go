package llmgo

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewTokenizer(t *testing.T) {
	tokenizer, err := NewTokenizer("./gpt2_tokenizer.bin")
	assert.NoError(t, err)
	orig := "input"
	encoded, err := tokenizer.Encode(orig)
	assert.NoError(t, err)
	decoded, err := tokenizer.Decode(encoded)
	assert.NoError(t, err)
	assert.Equal(t, orig, decoded)

}
