package main

import (
	"encoding/binary"
	"errors"
	"io"
	"os"
)

const Int32ByteLen = 4

type DataLoader struct {
	filename        string
	batchSize       int
	seqLength       int
	file            io.Reader
	currentPosition int64
	fileSize        int64
	buffer          []byte // Assuming tokenized data is represented as int32
	numBatches      int
	data            []int32
	dataAll         []int32
}

func NewDataLoader(filename string, batchSize, seqLength int) (*DataLoader, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, err
	}
	return newDataLoader(file, batchSize, seqLength, int(fileInfo.Size()))
}

func newDataLoader(file io.Reader, batchSize, seqLength, size int) (*DataLoader, error) {
	if size < (batchSize*seqLength+1)*Int32ByteLen {
		return nil, errors.New("error: file size is too small for the batch size and sequence length")
	}
	loader := &DataLoader{
		batchSize:  batchSize,
		seqLength:  seqLength,
		file:       file,
		numBatches: size / (batchSize * seqLength * Int32ByteLen),
		data:       make([]int32, size/Int32ByteLen),
		fileSize:   int64(size / Int32ByteLen),
	}
	if err := binary.Read(loader.file, binary.LittleEndian, loader.data); err != nil {
		return nil, err
	}
	return loader, nil
}

func (loader *DataLoader) Reset() {
	loader.currentPosition = 0
}

func (loader *DataLoader) NextBatch() ([]int32, []int32, error) {
	nextPos := loader.currentPosition + int64(loader.batchSize*loader.seqLength)
	if nextPos+1 > loader.fileSize {
		loader.Reset()
		nextPos = loader.currentPosition + int64(loader.batchSize*loader.seqLength)
	}
	// don't  x4 because we're indexing int32 not byte
	inputs := loader.data[loader.currentPosition:nextPos]
	targets := loader.data[loader.currentPosition+1 : nextPos+1]
	loader.currentPosition = nextPos
	return inputs, targets, nil
}
