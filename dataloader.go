package llmgo

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
)

const Int32ByteLen = 4

type DataLoader struct {
	filename        string
	batchSize       int
	seqLength       int
	currentPosition int64
	fileSize        int64
	NumBatches      int
	data            []int32
	dataAll         []int32
}

func NewDataLoader(filename string, batchSize, seqLength int) (*DataLoader, error) {
	file, err := Open(filename)
	if err != nil {
		return nil, err
	}
	return newDataLoader(file, batchSize, seqLength)
}

func newDataLoader(file io.Reader, batchSize, seqLength int) (*DataLoader, error) {
	data, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}
	size := len(data)
	if size < (batchSize*seqLength+1)*Int32ByteLen {
		return nil, errors.New("error: file size is too small for the batch size and sequence length")
	}
	loader := &DataLoader{
		batchSize:  batchSize,
		seqLength:  seqLength,
		NumBatches: size / (batchSize * seqLength * Int32ByteLen),
		data:       make([]int32, size/Int32ByteLen),
		fileSize:   int64(size / Int32ByteLen),
	}
	if err := binary.Read(bytes.NewReader(data), binary.LittleEndian, loader.data); err != nil {
		return nil, err
	}
	return loader, nil
}

func newDataLoaderFromInts(data []int32, batchSize, seqLength int) (*DataLoader, error) {
	size := len(data)
	if size < (batchSize*seqLength + 1) {
		return nil, errors.New("error: file size is too small for the batch size and sequence length")
	}
	loader := &DataLoader{
		batchSize:  batchSize,
		seqLength:  seqLength,
		NumBatches: size / (batchSize * seqLength),
		data:       data,
		fileSize:   int64(size),
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
