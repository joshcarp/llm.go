package main

import (
	"bytes"
	"encoding/binary"
	"io"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDataLoader_NextBatch(t *testing.T) {
	zeroTo100 := []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99}
	type want struct {
		reset           bool
		input           []int32
		target          []int32
		currentPosition int64
	}
	tests := []struct {
		name              string
		contents          []int32
		filename          string
		batchSize, seqLen int
		want              []want
		wantNumBatches    int
	}{
		{
			name:           "1char",
			contents:       []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
			batchSize:      1,
			seqLen:         1,
			wantNumBatches: 10,
			want: []want{
				{
					input:           []int32{0},
					target:          []int32{1},
					currentPosition: 1,
				},
				{
					input:           []int32{1},
					target:          []int32{2},
					currentPosition: 2,
				},
				{
					reset:           true,
					input:           []int32{0},
					target:          []int32{1},
					currentPosition: 1,
				},
			},
		},
		{
			name:           "endOfFile",
			contents:       []int32{0, 1, 2},
			batchSize:      1,
			seqLen:         1,
			wantNumBatches: 3,
			want: []want{
				{
					input:           []int32{0},
					target:          []int32{1},
					currentPosition: 1,
				},
				{
					input:           []int32{1},
					target:          []int32{2},
					currentPosition: 2,
				},
				{ // should loop back
					input:           []int32{0},
					target:          []int32{1},
					currentPosition: 1,
				},
				{
					reset:           true,
					input:           []int32{0},
					target:          []int32{1},
					currentPosition: 1,
				},
			},
		},
		{
			name:           "seqLen4",
			contents:       []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
			batchSize:      1,
			seqLen:         4,
			wantNumBatches: 2,
			want: []want{
				{
					input:           []int32{0, 1, 2, 3},
					target:          []int32{1, 2, 3, 4},
					currentPosition: 4,
				},
				{
					input:           []int32{4, 5, 6, 7},
					target:          []int32{5, 6, 7, 8},
					currentPosition: 8,
				},
			},
		},
		{
			name:           "seqLen!=batchSize",
			contents:       zeroTo100,
			batchSize:      2,
			seqLen:         4,
			wantNumBatches: 12,
			want: []want{
				{
					input:           []int32{0, 1, 2, 3, 4, 5, 6, 7},
					target:          []int32{1, 2, 3, 4, 5, 6, 7, 8},
					currentPosition: 8,
				},
				{
					input:           []int32{8, 9, 10, 11, 12, 13, 14, 15},
					target:          []int32{9, 10, 11, 12, 13, 14, 15, 16},
					currentPosition: 16,
				},
				{
					reset:           true,
					input:           []int32{0, 1, 2, 3, 4, 5, 6, 7},
					target:          []int32{1, 2, 3, 4, 5, 6, 7, 8},
					currentPosition: 8,
				},
			},
		},
	}
	newInt32Reader := func(data []int32) (io.Reader, int) {
		var b bytes.Buffer
		require.NoError(t, binary.Write(&b, binary.LittleEndian, data))
		return &b, b.Len()
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader, size := newInt32Reader(tt.contents)
			if tt.filename != "" {
				fileInfo, err := os.Stat(tt.filename)
				assert.NoError(t, err)
				file, err := os.Open(tt.filename)
				assert.NoError(t, err)
				defer file.Close()
				reader = file
				size = int(fileInfo.Size())

			}
			loader, err := newDataLoader(reader, tt.batchSize, tt.seqLen, size)
			assert.NoError(t, err)
			assert.Equal(t, tt.wantNumBatches, loader.numBatches)
			for _, want := range tt.want {
				if want.reset {
					loader.Reset()
				}
				input, target, err := loader.NextBatch()
				assert.NoError(t, err)
				assert.Equal(t, want.input, input)
				assert.Equal(t, want.target, target)
				assert.Equal(t, want.currentPosition, loader.currentPosition)
			}
		})
	}
}
