package main

import (
	"encoding/binary"
	"errors"
	"os"
)

type Tokenizer struct {
	vocabSize  uint32
	tokenTable []string
	init       bool
}

func NewTokenizer(filename string) (Tokenizer, error) {
	f, err := os.Open(filename)
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
	}
	return tok, nil
}

func (t Tokenizer) Decode(tokens []int32) (string, error) {
	s := ""
	for _, token := range tokens {
		if token >= int32(len(t.tokenTable)) {
			return "", errors.New("not valid token")
		}
		s += t.tokenTable[token]
	}
	return s, nil
}
