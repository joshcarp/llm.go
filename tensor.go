package main

type tensor[T any] struct {
	data []T
	dims []int
}

func newTensor[T any](data []T, dims ...int) (tensor[T], int) {
	s := 1
	for _, d := range dims {
		s *= d
	}
	if s > len(data) {
		panic("here")
	}
	ss := min(s, len(data))
	return tensor[T]{
		data: data[:ss],
		dims: dims,
	}, ss
}

func flatTensor[T any](data []T) tensor[T] {
	return tensor[T]{
		data: data,
		dims: []int{len(data)},
	}
}

func (t tensor[T]) size() int {
	size := 1
	for _, dim := range t.dims {
		size *= dim
	}
	return size
}

func (t tensor[T]) index(idx ...int) tensor[T] {
	// 1. Error Handling (Partially Adjusted)
	if len(idx) > len(t.dims) {
		panic("Too many indices for tensor dimensions")
	}
	for i, dim := range idx {
		if dim < 0 || dim >= t.dims[i] {
			panic("Index out of bounds")
		}
	}

	// 2. Calculate Linear Index (Partially Adjusted)
	linearIndex := idx[0]
	stride := t.size()
	for i := 1; i < len(idx); i++ {
		stride /= t.dims[i]
		linearIndex += idx[i] * stride
	}

	// 3. Adjust Dimensions and Return Sub-Tensor
	newDims := t.dims[len(idx):]                  // Keep remaining dimensions
	end := linearIndex + t.subTensorSize(newDims) // Size based on remaining dimensions

	return tensor[T]{
		data: t.data[linearIndex:end],
		dims: newDims,
	}
}

// Helper function to calculate the size of a sub-tensor
func (t tensor[T]) subTensorSize(idx []int) int {
	subTensorSize := 1
	for _, dim := range t.dims[len(idx):] {
		subTensorSize *= dim
	}
	return subTensorSize
}

// ParameterTensors
type ParameterTensors struct {
	WordTokEmbed tensor[float32] // (V, C) - Word/Token Embedding weights (Vocabulary size, Embedding dimension)
	WordPosEmbed tensor[float32] // (maxT, C) - Positional Embedding weights (Maximum Sequence length, Embedding dimension)

	Layer1NormW tensor[float32] // (L, C) - Layer Normalization weights for layer 1 (Number of layers, Embedding dimension)
	Layer1NormB tensor[float32] // (L, C) - Layer Normalization biases for layer 1

	QueryKeyValW tensor[float32] // (L, 3*C, C) - Attention QKV weights (Layers, 3 * Embedding dimension, Embedding dimension)
	QueryKeyValB tensor[float32] // (L, 3*C) - Attention QKV biases

	AttProjW tensor[float32] // (L, C, C) - Attention projection weights (Layers, Embedding dimension, Embedding dimension)
	AttProjB tensor[float32] // (L, C) - Attention projection biases

	Layer2NormW tensor[float32] // (L, C) - Layer Normalization weights for layer 2
	Layer2NormB tensor[float32] // (L, C) - Layer Normalization biases for layer 2

	FeedFwdW tensor[float32] // (L, 4*C, C) - Feed-forward layer weights (Layers, 4 * Embedding Dimension, Embedding Dimension)
	FeedFwdB tensor[float32] // (L, 4*C) - Feed-forward layer biases

	FeedFwdProjW tensor[float32] // (L, C, 4*C) - Feed-forward projection weights
	FeedFwdProjB tensor[float32] // (L, C)- Feed-forward projection biases

	LayerFinNormW tensor[float32] // (C) - Final layer normalization weights
	LayerFinNormB tensor[float32] // (C) - Final layer normalization biases
}

func (tensor *ParameterTensors) init(vocabSize, channels, maxSeqLen, numLayers int) []float32 {
	// So here basically we need to know what the size of each tensor so that we can init a contiguous slice
	memory := make([]float32,
		vocabSize*channels+
			maxSeqLen*channels+
			numLayers*channels+
			numLayers*channels+
			numLayers*3*channels*channels+
			numLayers*3*channels+
			numLayers*channels*channels+
			numLayers*channels+
			numLayers*channels+
			numLayers*channels+
			numLayers*4*channels*channels+
			numLayers*4*channels+
			numLayers*channels*4*channels+
			numLayers*channels+
			channels+
			channels)
	var ptr int
	memPtr := memory
	tensor.WordTokEmbed, ptr = newTensor(memPtr, vocabSize, channels)
	memPtr = memPtr[ptr:]
	tensor.WordPosEmbed, ptr = newTensor(memPtr, maxSeqLen, channels)
	memPtr = memPtr[ptr:]
	tensor.Layer1NormW, ptr = newTensor(memPtr, numLayers, channels)
	memPtr = memPtr[ptr:]
	tensor.Layer1NormB, ptr = newTensor(memPtr, numLayers, channels)
	memPtr = memPtr[ptr:]
	tensor.QueryKeyValW, ptr = newTensor(memPtr, numLayers, 3*channels, channels)
	memPtr = memPtr[ptr:]
	tensor.QueryKeyValB, ptr = newTensor(memPtr, numLayers, 3*channels)
	memPtr = memPtr[ptr:]
	tensor.AttProjW, ptr = newTensor(memPtr, numLayers, channels, channels)
	memPtr = memPtr[ptr:]
	tensor.AttProjB, ptr = newTensor(memPtr, numLayers, channels)
	memPtr = memPtr[ptr:]
	tensor.Layer2NormW, ptr = newTensor(memPtr, numLayers, channels)
	memPtr = memPtr[ptr:]
	tensor.Layer2NormB, ptr = newTensor(memPtr, numLayers, channels)
	memPtr = memPtr[ptr:]
	tensor.FeedFwdW, ptr = newTensor(memPtr, numLayers, 4*channels, channels)
	memPtr = memPtr[ptr:]
	tensor.FeedFwdB, ptr = newTensor(memPtr, numLayers, 4*channels)
	memPtr = memPtr[ptr:]
	tensor.FeedFwdProjW, ptr = newTensor(memPtr, numLayers, channels, 4*channels)
	memPtr = memPtr[ptr:]
	tensor.FeedFwdProjB, ptr = newTensor(memPtr, numLayers, channels)
	memPtr = memPtr[ptr:]
	tensor.LayerFinNormW, ptr = newTensor(memPtr, channels)
	memPtr = memPtr[ptr:]
	tensor.LayerFinNormB, ptr = newTensor(memPtr, channels)
	memPtr = memPtr[ptr:]
	if len(memPtr) != 0 {
		panic("something went real bad here")
	}
	return memory
}

// ActivationTensors
type ActivationTensors struct {
	// Encoded initially has the simple word token embeddings of the model.
	Encoded tensor[float32] // (B, T, C) - Initial encoded input representations (Batch size, Sequence length, Embedding dimension)

	Layer1Act tensor[float32] // (L, B, T, C) - Activations after Layer Normalization 1
	Ln1Mean   tensor[float32] // (L, B, T) - Mean values for Layer Normalization 1
	Ln1Rstd   tensor[float32] // (L, B, T) - Reciprocal of standard deviation for Layer Normalization 1

	Qkv     tensor[float32] // (L, B, T, 3*C) - Combined Query, Key, Value representations for attention
	Atty    tensor[float32] // (L, B, T, C) - Intermediate attention-like result
	Preatt  tensor[float32] // (L, B, NH, T, T) - Pre-attention scores (likely before softmax)
	Att     tensor[float32] // (L, B, NH, T, T) - Normalized attention weights (Number of layers, Batch size, Number of Attention Heads, Sequence length, Sequence length)
	Attproj tensor[float32] // (L, B, T, C) - Projected attention outputs

	Residual2 tensor[float32] // (L, B, T, C) - Residual connection after attention
	Ln2       tensor[float32] // (L, B, T, C) - Activations after Layer Normalization 2
	Ln2Mean   tensor[float32] // (L, B, T) - Mean values for Layer Normalization 2
	Ln2Rstd   tensor[float32] // (L, B, T) - Reciprocal of standard deviation for Layer Normalization 2

	Fch     tensor[float32] // (L, B, T, 4*C) - Intermediate Feed-Forward Network activations
	FchGelu tensor[float32] // (L, B, T, 4*C) - Fch activations after applying GELU (non-linearity)
	Fcproj  tensor[float32] // (L, B, T, C) - Projected output of the Feed-Forward Network

	Residual3 tensor[float32] // (L, B, T, C) - Residual connection after Feed-Forward Network
	Lnf       tensor[float32] // (B, T, C) - Final activations after Layer Normalization
	LnfMean   tensor[float32] // (B, T) - Mean values for final Layer Normalization
	LnfRstd   tensor[float32] // (B, T) - Reciprocal of standard deviation for final Layer Normalization

	Logits tensor[float32] // (B, T, V) - Raw output scores (before softmax)
	Probs  tensor[float32] // (B, T, V) - Softmax probabilities over the vocabulary
	Losses tensor[float32] // (B, T) - Loss values per token in the batch
}

func (tensor *ActivationTensors) init(B, C, T, L, NH, V int) []float32 {
	memory := make([]float32,
		B*T*C+
			L*B*T*C+
			L*B*T+
			L*B*T+
			L*B*T*C*3+
			L*B*T*C+
			L*B*NH*T*T+
			L*B*NH*T*T+
			L*B*T*C+
			L*B*T*C+
			L*B*T*C+
			L*B*T+
			L*B*T+
			L*B*T*C*4+
			L*B*T*C*4+
			L*B*T*C+
			L*B*T*C+
			B*T*C+
			B*T+
			B*T+
			B*T*V+
			B*T*V+
			B*T)
	var ptr int
	memPtr := memory
	tensor.Encoded, ptr = newTensor(memPtr, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Layer1Act, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Ln1Mean, ptr = newTensor(memPtr, L, B, T)
	memPtr = memPtr[ptr:]
	tensor.Ln1Rstd, ptr = newTensor(memPtr, L, B, T)
	memPtr = memPtr[ptr:]
	tensor.Qkv, ptr = newTensor(memPtr, L, B, T, C*3)
	memPtr = memPtr[ptr:]
	tensor.Atty, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Preatt, ptr = newTensor(memPtr, L, B, NH, T, T)
	memPtr = memPtr[ptr:]
	tensor.Att, ptr = newTensor(memPtr, L, B, NH, T, T)
	memPtr = memPtr[ptr:]
	tensor.Attproj, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Residual2, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Ln2, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Ln2Mean, ptr = newTensor(memPtr, L, B, T)
	memPtr = memPtr[ptr:]
	tensor.Ln2Rstd, ptr = newTensor(memPtr, L, B, T)
	memPtr = memPtr[ptr:]
	tensor.Fch, ptr = newTensor(memPtr, L, B, T, C*4)
	memPtr = memPtr[ptr:]
	tensor.FchGelu, ptr = newTensor(memPtr, L, B, T, C*4)
	memPtr = memPtr[ptr:]
	tensor.Fcproj, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Residual3, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Lnf, ptr = newTensor(memPtr, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.LnfMean, ptr = newTensor(memPtr, B, T)
	memPtr = memPtr[ptr:]
	tensor.LnfRstd, ptr = newTensor(memPtr, B, T)
	memPtr = memPtr[ptr:]
	tensor.Logits, ptr = newTensor(memPtr, B, T, V)
	memPtr = memPtr[ptr:]
	tensor.Probs, ptr = newTensor(memPtr, B, T, V)
	memPtr = memPtr[ptr:]
	tensor.Losses, ptr = newTensor(memPtr, B, T)
	memPtr = memPtr[ptr:]
	if len(memPtr) != 0 {
		panic("something went real bad here")
	}
	return memory
}
