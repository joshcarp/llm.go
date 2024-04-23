package llmgo

import (
	"encoding/binary"
	"errors"
)

type Tokenizer struct {
	vocabSize  uint32
	tokenTable []string
	trie       trie
	init       bool
}

func newTokenizer(vocab []string) Tokenizer {
	tokenizer := Tokenizer{
		vocabSize:  uint32(len(vocab)),
		tokenTable: vocab,
		trie:       newTrie(vocab),
		init:       true,
	}
	return tokenizer
}
func NewTokenizer(filename string) (Tokenizer, error) {
	f, err := Open(filename)
	if err != nil {
		return Tokenizer{}, err
	}
	defer f.Close()
	header := make([]uint32, 256)
	if err := binary.Read(f, binary.LittleEndian, header); err != nil {
		return Tokenizer{}, err
	}
	if header[0] != 20240328 || header[1] != 1 {
		return Tokenizer{}, errors.New("incorrect header for tokenizer")
	}
	tok := Tokenizer{
		vocabSize:  header[2],
		tokenTable: make([]string, header[2]),
		init:       true,
		trie:       newTrie(nil),
	}
	var length byte
	for i := range tok.tokenTable {
		if err := binary.Read(f, binary.LittleEndian, &length); err != nil {
			return tok, err
		}
		if length <= 0 {
			return tok, errors.New("tokenizer failure")
		}
		tokenBytes := make([]byte, length)
		if err := binary.Read(f, binary.LittleEndian, tokenBytes); err != nil {
			return tok, err
		}
		tok.tokenTable[i] = string(tokenBytes)
		tok.trie.Insert(tokenBytes, int32(i))
	}
	return tok, nil
}

func (t Tokenizer) Decode(tokens []int32) (string, error) {
	s := ""
	for _, token := range tokens {
		if token >= int32(len(t.tokenTable)) {
			return "", errors.New("not valid token")
		}
		if token != GPT2_EOT {
			s += t.tokenTable[token]
		}
	}
	return s, nil
}

func (t Tokenizer) Encode(text string) ([]int32, error) {
	_, tokens := t.trie.Tokenize([]byte(text))
	return tokens, nil
}

type trie struct {
	children map[byte]*trie
	data     int32
	end      bool
	key      byte
}

func newTrie(data []string) trie {
	t := trie{
		children: map[byte]*trie{},
		end:      false,
	}
	for i, word := range data {
		t.Insert([]byte(word), int32(i))
	}
	return t
}

func (t *trie) Insert(word []byte, data int32) error {
	cur := t
	if len(word) == 0 {
		return errors.New("zero length word not supported")
	}
	var index byte
	for i := 0; i < len(word); i++ {
		index = word[i] // 00: 0
		if cur.children[index] == nil {
			cur.children[index] = &trie{
				children: map[byte]*trie{},
			}
		}
		cur = cur.children[index]
	}
	cur.end = true
	cur.data = data
	cur.key = index
	return nil
}

func (t *trie) Tokenize(input []byte) ([][]byte, []int32) {
	var cur = t
	var token = GPT2_EOT
	endIdx, next := 1, 0
	split, tokens := make([][]byte, 0), make([]int32, 0)
	for len(input) != 0 {
		switch {
		case next == len(input), cur.children[input[next]] == nil:
			split = append(split, input[:endIdx])
			tokens = append(tokens, token)
			input = input[endIdx:]
			token = GPT2_EOT
			cur = t
			next = 0
			endIdx = 1
		default:
			cur = cur.children[input[next]]
			next += 1
			if cur.end {
				endIdx = next
				token = cur.data
			}
		}
	}
	return split, tokens
}
