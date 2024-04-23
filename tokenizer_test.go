package llmgo

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	rand "math/rand"
	"sort"
	"strings"
	"testing"
)

func FuzzNewTokenizer(f *testing.F) {
	tokenizer, err := NewTokenizer("./gpt2_tokenizer.bin")
	assert.NoError(f, err)
	// Initialize the fuzz test inputs
	f.Fuzz(func(t *testing.T, orig string) {
		encoded, err := tokenizer.Encode(string(orig))
		assert.NoError(t, err)
		decoded, err := tokenizer.Decode(encoded)
		assert.NoError(t, err)
		assert.Equal(t, string(orig), decoded)
	})
}

func TestTokenizer(t *testing.T) {
	text := "000000000000000000000000000000"
	//text := "wc"
	println(text)
	tokenizer, err := NewTokenizer("./gpt2_tokenizer.bin")
	assert.NoError(t, err)
	encoded, err := tokenizer.Encode(text)
	fmt.Println(encoded)
	for _, tok := range encoded {
		decoded, err := tokenizer.Decode([]int32{tok})
		assert.NoError(t, err)
		print(decoded)
		print(", ")
	}
	assert.NoError(t, err)
	decoded, err := tokenizer.Decode(encoded)
	assert.NoError(t, err)
	assert.Equal(t, text, decoded)

}

func randomString(n int) string {
	chars := "abcdefghijklmnopqrstuvwxyz"
	var sb strings.Builder
	for i := 0; i < n; i++ {
		sb.WriteByte(chars[rand.Intn(len(chars))])
	}
	return sb.String()
}

func TestStringRange(t *testing.T) {
	str := "¥"
	println("len string is ", len(str))
	println("for _, s := range str {")
	for _, s := range str {
		println(s) // prints 165
	}
	println("for i := 0; i < len(str); i++ {")
	for i := 0; i < len(str); i++ {
		println(str[i]) // prints 194 then 165
	}
}

func TestNewTrie(t *testing.T) {
	tests := []struct {
		name       string
		words      []string
		input      string
		wantSplit  [][]byte
		wantTokens []int32
		wantErr    bool
	}{
		{
			name:       "",
			words:      []string{"a", "b"},
			input:      "ab",
			wantSplit:  [][]byte{[]byte("a"), []byte("b")},
			wantTokens: []int32{0, 1},
		},
		{
			name:       "",
			words:      []string{"These", "are", "some", "words", " "},
			input:      "These are some words",
			wantSplit:  [][]byte{[]byte("These"), []byte(" "), []byte("are"), []byte(" "), []byte("some"), []byte(" "), []byte("words")},
			wantTokens: []int32{0, 4, 1, 4, 2, 4, 3},
		},
		{
			name:       "UnknownToken",
			words:      []string{"a", "b"},
			input:      "abc",
			wantSplit:  [][]byte{[]byte("a"), []byte("b"), []byte("c")},
			wantTokens: []int32{0, 1, GPT2_EOT},
		},
		{
			name:       "UnknownToken2",
			words:      []string{"a", "b"},
			input:      "acb",
			wantSplit:  [][]byte{[]byte("a"), []byte("c"), []byte("b")},
			wantTokens: []int32{0, GPT2_EOT, 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr := newTrie(tt.words)
			split, tokens := tr.Tokenize([]byte(tt.input))
			assert.Equal(t, tt.wantSplit, split)
			assert.Equal(t, tt.wantTokens, tokens)
		})
	}
}

// FuzzTrie is a very inefficient way of fuzz testing a trie but it works
func FuzzTrie(f *testing.F) {
	byteSlice := func(data []byte, dimensions []byte) [][]byte {
		var result [][]byte
		var index int
		seen := make(map[string]bool)

		for _, dim := range dimensions {
			size := int(dim) + 1
			if index+size > len(data) {
				return result
			}
			str := string(data[index : index+size])
			if !seen[str] && len(str) > 0 {
				result = append(result, data[index:index+size])
				seen[str] = true
			}
			index += size
			if index >= len(data) {
				return result
			}
		}
		return result
	}
	breakIntoTokens := func(input []byte, tokensUnsorted []string) [][]byte {
		tokens := make([]string, len(tokensUnsorted))
		copy(tokens, tokensUnsorted)
		sort.Slice(tokens, func(i, j int) bool {
			return len(tokens[i]) > len(tokens[j])
		})
		result := make([][]byte, 0)
		i := 0
		for i < len(input) {
			var longest string
			for j := i + 1; j <= len(input); j++ {
				substring := input[i:j]
				for _, token := range tokens {
					if string(substring) == token {
						longest = token
					}
				}
			}
			if len(longest) > 0 {
				result = append(result, []byte(longest))
				i += len(longest)
			} else {
				result = append(result, input[i:i+1])
				i += 1
			}
		}
		return result
	}
	f.Fuzz(func(t *testing.T, vocabBytes, vocabDims, input []byte) {
		vocab := byteSlice(vocabBytes, vocabDims)
		// In this test the `input` slice are the tokens we want to get back.
		// We concatenate them into a single string and assert that the tokenizer gives them back.
		wantTokens := make([]int32, 0)
		vocabMap := make(map[string]int32, len(vocab))
		vocabStrings := make([]string, 0, len(vocab))
		for i, key := range vocab {
			vocabMap[string(key)] = int32(i)
			vocabStrings = append(vocabStrings, string(key)) // 0: 0, 00: 1
		}
		wantSplit := breakIntoTokens(input, vocabStrings)
		for _, str := range wantSplit {
			tok, ok := vocabMap[string(str)]
			if !ok {
				wantTokens = append(wantTokens, GPT2_EOT)
			} else {
				wantTokens = append(wantTokens, tok)
			}
		}
		tr := newTrie(vocabStrings)
		split, tokens := tr.Tokenize(input)
		assert.Equal(t, wantSplit, split)
		assert.Equal(t, wantTokens, tokens)
	})
}
