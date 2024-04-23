package llmgo

import (
	"encoding/binary"
	"errors"
	"fmt"
)

type Tokenizer struct {
	vocabSize   uint32
	tokenTable  []string
	lookupTable map[string]int32
	init        bool
}

func NewTokenizer(filename string) (Tokenizer, error) {
	fmt.Println("NewTokenizer", filename)
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
		vocabSize:   header[2],
		tokenTable:  make([]string, header[2]),
		lookupTable: make(map[string]int32, header[2]),
		init:        true,
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
		tok.lookupTable[string(tokenBytes)] = int32(i)
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
	trie := NewTrie()
	for _, token := range t.tokenTable {
		trie.Insert(token)
	}
	tokensStrings, tokenIDs := []string{}, []int32{}
	currentToken := ""
	for _, s := range text {
		if !trie.StartsWith(currentToken + string(s)) {
			tokensStrings = append(tokensStrings, currentToken)
			currentToken = string(s)
		} else {
			currentToken += string(s)
		}
	}
	if currentToken != "" {
		tokensStrings = append(tokensStrings, currentToken)
	}
	for _, str := range tokensStrings {
		val, ok := t.lookupTable[str]
		if !ok {
			continue
			panic(fmt.Sprintf("token not found: %s", str))
		}
		tokenIDs = append(tokenIDs, val)
	}
	return tokenIDs, nil
}

type Trie struct {
	children map[string]*Trie
	end      bool
}

func NewTrie() Trie {
	return Trie{
		children: map[string]*Trie{},
		end:      false,
	}
}

func (this *Trie) Insert(word string) {
	cur := this
	for i := 0; i < len(word); i++ {
		index := word[i]
		if cur.children[string(index)] == nil {
			cur.children[string(index)] = &Trie{
				children: map[string]*Trie{},
			}
		}
		cur = cur.children[string(index)]
	}
	cur.end = true
}

func (this *Trie) Search(word string) bool {
	cur := this
	for i := 0; i < len(word); i++ {
		index := word[i] - 'a'
		if cur.children[string(index)] == nil {
			return false
		}
		cur = cur.children[string(index)]
	}

	return cur.end
}

func (this *Trie) StartsWith(prefix string) bool {
	cur := this
	for i := 0; i < len(prefix); i++ {
		index := prefix[i]
		if cur.children[string(index)] == nil {
			return false
		}
		cur = cur.children[string(index)]
	}
	return true
}
