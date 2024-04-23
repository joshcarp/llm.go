package llmgo

type tensor struct {
	data []float32
	dims []int
}

// TODO: make this better
func (t tensor) Data() []float32 {
	return t.data
}

func newTensor(data []float32, dims ...int) (tensor, int) {
	s := 1
	for _, d := range dims {
		s *= d
	}
	if s > len(data) {
		panic("dimensions larger than supplied data")
	}
	ss := min(s, len(data))
	return tensor{
		data: data[:ss],
		dims: dims,
	}, ss
}

func (t tensor) size() int {
	size := 1
	for _, dim := range t.dims {
		size *= dim
	}
	return size
}

func (t tensor) index(idx ...int) tensor {
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

	return tensor{
		data: t.data[linearIndex:end],
		dims: newDims,
	}
}

// Helper function to calculate the size of a sub-tensor
func (t tensor) subTensorSize(idx []int) int {
	subTensorSize := 1
	for _, dim := range t.dims[len(idx):] {
		subTensorSize *= dim
	}
	return subTensorSize
}

// ParameterTensors are the parameters of the model
type ParameterTensors struct {
	Memory        []float32
	WordTokEmbed  tensor // (V, C) - Word/Token Embedding weights (Vocabulary size, Embedding dimension)
	WordPosEmbed  tensor // (maxT, C) - Positional Embedding weights (Maximum Sequence length, Embedding dimension)
	LayerNorm1W   tensor // (L, C) - Weights for Layer Normalization 1 (Number of layers, Embedding dimension)
	LayerNorm1B   tensor // (L, C) - Biases for Layer Normalization 1
	QueryKeyValW  tensor // (L, 3*C, C) - Attention QKV weights (Layers, 3 * Embedding dimension, Embedding dimension)
	QueryKeyValB  tensor // (L, 3*C) - Attention QKV biases
	AttProjW      tensor // (L, C, C) - Attention projection weights (Layers, Embedding dimension, Embedding dimension)
	AttProjB      tensor // (L, C) - Attention projection biases
	Layer2NormW   tensor // (L, C) - Weights for Layer Normalization 2
	Layer2NormB   tensor // (L, C) - Biases for Layer Normalization 2
	FeedFwdW      tensor // (L, 4*C, C) - Feed-forward layer weights (Layers, 4 * Embedding Dimension, Embedding Dimension)
	FeedFwdB      tensor // (L, 4*C) - Feed-forward layer biases
	FeedFwdProjW  tensor // (L, C, 4*C) - Feed-forward projection weights
	FeedFwdProjB  tensor // (L, C)- Feed-forward projection biases
	LayerFinNormW tensor // (C) - Final layer normalization weights
	LayerFinNormB tensor // (C) - Final layer normalization biases
}

// Init initialises the ParameterTensors with specific sizes for each tensor based on the model architecture.
func (tensor *ParameterTensors) Init(V, C, maxSeqLen, L int) {
	tensor.Memory = make([]float32,
		V*C+ // WordTokEmbed
			maxSeqLen*C+ // WordPosEmbed
			L*C+ // LayerNorm1W
			L*C+ // LayerNorm1B
			L*3*C*C+ // QueryKeyValW
			L*3*C+ // QueryKeyValB
			L*C*C+ // AttProjW
			L*C+ // AttProjB
			L*C+ // Layer2NormW
			L*C+ // Layer2NormB
			L*4*C*C+ // FeedFwdW
			L*4*C+ // FeedFwdB
			L*C*4*C+ // FeedFwdProjW
			L*C+ // FeedFwdProjB
			C+ // LayerFinNormW
			C, // LayerFinNormB
	)
	var ptr int
	memPtr := tensor.Memory
	tensor.WordTokEmbed, ptr = newTensor(memPtr, V, C)
	memPtr = memPtr[ptr:]
	tensor.WordPosEmbed, ptr = newTensor(memPtr, maxSeqLen, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm1W, ptr = newTensor(memPtr, L, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm1B, ptr = newTensor(memPtr, L, C)
	memPtr = memPtr[ptr:]
	tensor.QueryKeyValW, ptr = newTensor(memPtr, L, 3*C, C)
	memPtr = memPtr[ptr:]
	tensor.QueryKeyValB, ptr = newTensor(memPtr, L, 3*C)
	memPtr = memPtr[ptr:]
	tensor.AttProjW, ptr = newTensor(memPtr, L, C, C)
	memPtr = memPtr[ptr:]
	tensor.AttProjB, ptr = newTensor(memPtr, L, C)
	memPtr = memPtr[ptr:]
	tensor.Layer2NormW, ptr = newTensor(memPtr, L, C)
	memPtr = memPtr[ptr:]
	tensor.Layer2NormB, ptr = newTensor(memPtr, L, C)
	memPtr = memPtr[ptr:]
	tensor.FeedFwdW, ptr = newTensor(memPtr, L, 4*C, C)
	memPtr = memPtr[ptr:]
	tensor.FeedFwdB, ptr = newTensor(memPtr, L, 4*C)
	memPtr = memPtr[ptr:]
	tensor.FeedFwdProjW, ptr = newTensor(memPtr, L, C, 4*C)
	memPtr = memPtr[ptr:]
	tensor.FeedFwdProjB, ptr = newTensor(memPtr, L, C)
	memPtr = memPtr[ptr:]
	tensor.LayerFinNormW, ptr = newTensor(memPtr, C)
	memPtr = memPtr[ptr:]
	tensor.LayerFinNormB, ptr = newTensor(memPtr, C)
	memPtr = memPtr[ptr:]
	if len(memPtr) != 0 {
		panic("something went real bad here")
	}
}

// ActivationTensors
type ActivationTensors struct {
	Memory             []float32
	Encoded            tensor // (B, T, C) - Initial encoded input representations (Batch size, Sequence length, Embedding dimension)
	Layer1Act          tensor // (L, B, T, C) - Activations after Layer Normalization 1
	LayerNorm1Mean     tensor // (L, B, T) - Mean values for Layer Normalization 1
	LayerNorm1Rstd     tensor // (L, B, T) - Reciprocal of standard deviation for Layer Normalization 1
	QueryKeyVal        tensor // (L, B, T, 3*C) - Combined Query, Key, Value representations for attention
	AttentionInter     tensor // (L, B, T, C) - Intermediate attention-like result
	PreAttention       tensor // (L, B, NH, T, T) - Pre-attention scores
	Attention          tensor // (L, B, NH, T, T) - Normalized attention weights (Number of layers, Batch size, Number of Attention Heads, Sequence length, Sequence length)
	AttentionProj      tensor // (L, B, T, C) - Projected attention outputs
	Residual2          tensor // (L, B, T, C) - Residual connection after attention
	LayerNorm2Act      tensor // (L, B, T, C) - Activations after Layer Normalization 2
	LayerNorm2Mean     tensor // (L, B, T) - Mean values for Layer Normalization 2
	LayerNorm2Rstd     tensor // (L, B, T) - Reciprocal of standard deviation for Layer Normalization 2
	FeedForward        tensor // (L, B, T, 4*C) - Intermediate Feed-Forward Network activations
	FeedForwardGelu    tensor // (L, B, T, 4*C) - FeedForward activations after applying GELU (non-linearity)
	FeedForwardProj    tensor // (L, B, T, C) - Projected output of the Feed-Forward Network
	Residual3          tensor // (L, B, T, C) - Residual connection after Feed-Forward Network
	LayerNormFinal     tensor // (B, T, C) - Final activations after Layer Normalization
	LayerNormFinalMean tensor // (B, T) - Mean values for final Layer Normalization
	LayerNormFinalStd  tensor // (B, T) - Reciprocal of standard deviation for final Layer Normalization
	Logits             tensor // (B, T, V) - Raw output scores (before softmax)
	Probabilities      tensor // (B, T, V) - Softmax probabilities over the vocabulary
	Losses             tensor // (B, T) - Loss values per token in the batch
}

func (tensor *ActivationTensors) Init(B, C, T, L, NH, V int) {
	tensor.Memory = make([]float32,
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
	memPtr := tensor.Memory
	tensor.Encoded, ptr = newTensor(memPtr, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Layer1Act, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm1Mean, ptr = newTensor(memPtr, L, B, T)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm1Rstd, ptr = newTensor(memPtr, L, B, T)
	memPtr = memPtr[ptr:]
	tensor.QueryKeyVal, ptr = newTensor(memPtr, L, B, T, C*3)
	memPtr = memPtr[ptr:]
	tensor.AttentionInter, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.PreAttention, ptr = newTensor(memPtr, L, B, NH, T, T)
	memPtr = memPtr[ptr:]
	tensor.Attention, ptr = newTensor(memPtr, L, B, NH, T, T)
	memPtr = memPtr[ptr:]
	tensor.AttentionProj, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Residual2, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm2Act, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm2Mean, ptr = newTensor(memPtr, L, B, T)
	memPtr = memPtr[ptr:]
	tensor.LayerNorm2Rstd, ptr = newTensor(memPtr, L, B, T)
	memPtr = memPtr[ptr:]
	tensor.FeedForward, ptr = newTensor(memPtr, L, B, T, C*4)
	memPtr = memPtr[ptr:]
	tensor.FeedForwardGelu, ptr = newTensor(memPtr, L, B, T, C*4)
	memPtr = memPtr[ptr:]
	tensor.FeedForwardProj, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.Residual3, ptr = newTensor(memPtr, L, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNormFinal, ptr = newTensor(memPtr, B, T, C)
	memPtr = memPtr[ptr:]
	tensor.LayerNormFinalMean, ptr = newTensor(memPtr, B, T)
	memPtr = memPtr[ptr:]
	tensor.LayerNormFinalStd, ptr = newTensor(memPtr, B, T)
	memPtr = memPtr[ptr:]
	tensor.Logits, ptr = newTensor(memPtr, B, T, V)
	memPtr = memPtr[ptr:]
	tensor.Probabilities, ptr = newTensor(memPtr, B, T, V)
	memPtr = memPtr[ptr:]
	tensor.Losses, ptr = newTensor(memPtr, B, T)
	memPtr = memPtr[ptr:]
	if len(memPtr) != 0 {
		panic("something went real bad here")
	}
}
