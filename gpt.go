package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"time"
)

const GPT2_EOT = 50256

type GPT2Config struct {
	MaxSeqLen int `json:"max_seq_len"`
	VocabSize int `json:"vocab_size"`
	NumLayers int `json:"num_layers"`
	NumHeads  int `json:"num_heads"`
	Channels  int `json:"channels"`
}

type GPT2 struct {
	Config GPT2Config // Hyperparameters of the model

	// Params has the actual weights of the model. ParamsMemory is for convinience to be able to set/reset parameters simply
	Params       ParameterTensors // Weights of the model
	ParamsMemory []float32        // Potentially contiguous storage for parameters

	// Grads contains the delta/gradient that will eventually be applied to the params in the model
	Grads       ParameterTensors // Gradients of the weights
	GradsMemory []float32        // Potentially contiguous storage for gradients

	NumParameters int // Total number of parameters
	// Fields for AdamW optimizer
	MMemory    []float32         // First moment estimates (for AdamW)
	VMemory    []float32         // Second moment estimates (for AdamW)
	Acts       ActivationTensors // Activations of the model
	ActsMemory []float32         // Potentially contiguous storage for activations
	// gradients of the activations
	GradsActs       ActivationTensors
	NumActivations  int
	GradsActsMemory []float32
	BatchSize       int     // Current batch size (B)
	SeqLen          int     // Current sequence length (T)
	Inputs          []rune  // Input tokens
	Targets         []rune  // Target tokens
	MeanLoss        float32 // Mean loss after a forward pass
}

// LoadGPT2Model loads the GPT-2 model from a checkpoint file.
func LoadGPT2Model(checkpointPath string) (*GPT2, error) {
	// File Reading
	f, err := os.Open(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("Error opening model file: %v", err)
	}
	defer f.Close()
	// Read Model Header
	return loadFromReader(f)
}

func loadFromReader(f io.Reader) (*GPT2, error) {
	header := make([]int32, 256)
	err := binary.Read(f, binary.LittleEndian, header)
	if err != nil {
		return nil, fmt.Errorf("Error reading model header: %v", err)
	}
	if header[0] != 20240326 || header[1] != 1 {
		return nil, fmt.Errorf("Bad model file format")
	}
	model := &GPT2{
		Config: GPT2Config{
			MaxSeqLen: int(header[2]),
			VocabSize: int(header[3]),
			NumLayers: int(header[4]),
			NumHeads:  int(header[5]),
			Channels:  int(header[6]),
		},
	}
	model.ParamsMemory = model.Params.init(model.Config.VocabSize, model.Config.Channels, model.Config.MaxSeqLen, model.Config.NumLayers)
	model.NumParameters = len(model.ParamsMemory)
	if err := binary.Read(f, binary.LittleEndian, model.ParamsMemory); err != nil {
		return nil, fmt.Errorf("Error reading model: %v", err)
	}
	return model, nil
}

func (model *GPT2) String() string {
	var s string
	s += "[GPT-2]\n"
	s += fmt.Sprintf("max_seq_len: %d\n", model.Config.MaxSeqLen)
	s += fmt.Sprintf("vocab_size: %d\n", model.Config.VocabSize)
	s += fmt.Sprintf("num_layers: %d\n", model.Config.NumLayers)
	s += fmt.Sprintf("num_heads: %d\n", model.Config.NumHeads)
	s += fmt.Sprintf("channels: %d\n", model.Config.Channels)
	s += fmt.Sprintf("num_parameters: %d\n", model.NumParameters)
	return s
}

func (model *GPT2) update(learningRate, beta1, beta2, eps, weightDecay float32, t int) {
	// Lazy memory allocation
	if model.MMemory == nil {
		model.MMemory = make([]float32, model.NumParameters)
		model.VMemory = make([]float32, model.NumParameters)
	}
	// Parameter updates
	for i := 0; i < model.NumParameters; i++ {
		parameter := model.ParamsMemory[i]
		gradient := model.GradsMemory[i]
		// Momentum update
		m2 := beta1*model.MMemory[i] + (1.0-beta1)*gradient
		// RMSprop update
		v2 := beta2*model.VMemory[i] + (1.0-beta2)*gradient*gradient
		// Bias correction
		mHat := m2 / float32(1.0-math.Pow(float64(beta1), float64(t)))
		vHat := v2 / float32(1.0-math.Pow(float64(beta2), float64(t)))
		// Parameter update
		model.MMemory[i] = m2
		model.VMemory[i] = v2
		model.ParamsMemory[i] -= learningRate * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + weightDecay*parameter)
	}
}

func (model *GPT2) backward() {
	//// double check we forwarded previously, with targets
	if model.MeanLoss == -1.0 {
		fmt.Println("Error: must forward with targets before backward")
		os.Exit(1)
	}
	// lazily allocate the memory for gradients of the weights and activations, if needed
	// convenience shortcuts
	B := model.BatchSize
	T := model.SeqLen
	V := model.Config.VocabSize
	L := model.Config.NumLayers
	NH := model.Config.NumHeads
	C := model.Config.Channels
	if len(model.GradsMemory) == 0 {
		model.GradsMemory = model.Grads.init(model.Config.VocabSize, model.Config.Channels, model.Config.MaxSeqLen, model.Config.NumLayers)
		model.GradsActsMemory = model.GradsActs.init(B, C, T, L, NH, V)
		model.NumActivations = len(model.GradsActsMemory)
		model.zeroGradient()
	}

	// backward pass
	params := model.Params // for brevity
	grads := model.Grads   //
	acts := model.Acts
	gradsActs := model.GradsActs
	// we kick off the chain by filling in dlosses with 1.0f/(B*T), to get the mean loss
	dlossMean := float32(1.0 / float32(B*T))
	for i := range gradsActs.Losses.data {
		gradsActs.Losses.data[i] = dlossMean
	}
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)
	crossentropySoftmaxBackward(gradsActs.Logits.data, gradsActs.Losses.data, acts.Probs.data, model.Targets, B, T, V)
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)
	matmul_backward(gradsActs.Lnf.data, grads.WordTokEmbed.data, nil, gradsActs.Logits.data, acts.Lnf.data, params.WordTokEmbed.data, B, T, C, V)
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)
	residual := acts.Residual3.data[(L-1)*B*T*C:]       // last layer's residual
	dresidual := gradsActs.Residual3.data[(L-1)*B*T*C:] // write to last layer's residual
	layernormBackward(dresidual, grads.LayerFinNormW.data, grads.LayerFinNormB.data, gradsActs.Lnf.data, residual, params.LayerFinNormW.data, acts.LnfMean.data, acts.LnfRstd.data, B, T, C)
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)
	for l := L - 1; l >= 0; l-- {
		if l == 0 {
			residual = acts.Encoded.data
			dresidual = gradsActs.Encoded.data
		} else {
			residual = acts.Residual3.data[(l-1)*B*T*C:] // should be l not L
			dresidual = gradsActs.Residual3.data[(l-1)*B*T*C:]
		}

		// Assuming you have a 'params' variable of your ParameterTensors type
		l_ln1w := params.Layer1NormW.data[l*C:]
		l_qkvw := params.QueryKeyValW.data[l*3*C*C:]
		l_attprojw := params.AttProjW.data[l*C*C:]
		l_ln2w := params.Layer2NormW.data[l*C:]
		l_fcw := params.FeedFwdW.data[l*4*C*C:]
		l_fcprojw := params.FeedFwdProjW.data[l*C*4*C:]

		// Gradients of weights
		dl_ln1w := grads.Layer1NormW.data[l*C:]
		dl_ln1b := grads.Layer1NormB.data[l*C:]
		dl_qkvw := grads.QueryKeyValW.data[l*3*C*C:]
		dl_qkvb := grads.QueryKeyValB.data[l*3*C:]
		dl_attprojw := grads.AttProjW.data[l*C*C:]
		dl_attprojb := grads.AttProjB.data[l*C:]
		dl_ln2w := grads.Layer2NormW.data[l*C:]
		dl_ln2b := grads.Layer2NormB.data[l*C:]
		dl_fcw := grads.FeedFwdW.data[l*4*C*C:]
		dl_fcb := grads.FeedFwdB.data[l*4*C:]
		dl_fcprojw := grads.FeedFwdProjW.data[l*C*4*C:]
		dl_fcprojb := grads.FeedFwdProjB.data[l*C:]

		// Activations
		l_ln1 := acts.Layer1Act.data[l*B*T*C:]
		l_ln1_mean := acts.Ln1Mean.data[l*B*T:]
		l_ln1_rstd := acts.Ln1Rstd.data[l*B*T:]
		l_qkv := acts.Qkv.data[l*B*T*3*C:]
		l_atty := acts.Atty.data[l*B*T*C:]
		l_att := acts.Att.data[l*B*NH*T*T:]
		l_residual2 := acts.Residual2.data[l*B*T*C:]
		l_ln2 := acts.Ln2.data[l*B*T*C:]
		l_ln2_mean := acts.Ln2Mean.data[l*B*T:]
		l_ln2_rstd := acts.Ln2Rstd.data[l*B*T:]
		l_fch := acts.Fch.data[l*B*T*4*C:]
		l_fch_gelu := acts.FchGelu.data[l*B*T*4*C:]

		dl_ln1 := gradsActs.Layer1Act.data[l*B*T*C:]
		dl_qkv := gradsActs.Qkv.data[l*B*T*3*C:]
		dl_atty := gradsActs.Atty.data[l*B*T*C:]
		dl_preatt := gradsActs.Preatt.data[l*B*NH*T*T:]
		dl_att := gradsActs.Att.data[l*B*NH*T*T:]
		dl_attproj := gradsActs.Attproj.data[l*B*T*C:]
		dl_residual2 := gradsActs.Residual2.data[l*B*T*C:]
		dl_ln2 := gradsActs.Ln2.data[l*B*T*C:]
		dl_fch := gradsActs.Fch.data[l*B*T*4*C:]
		dl_fch_gelu := gradsActs.FchGelu.data[l*B*T*4*C:]
		dl_fcproj := gradsActs.Fcproj.data[l*B*T*C:]
		dl_residual3 := gradsActs.Residual3.data[l*B*T*C:]
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		residualBackward(dl_residual2, dl_fcproj, dl_residual3, B*T*C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		geluBackward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		layernormBackward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		residualBackward(dresidual, dl_attproj, dl_residual2, B*T*C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		attentionBackward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		layernormBackward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
	}
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)
	// Here we want to apply our gradients to our encoded data.
	encoderBackward(grads.WordTokEmbed.data, grads.WordPosEmbed.data, gradsActs.Encoded.data, model.Inputs, B, T, C)
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)
}

func (model *GPT2) train(valDataloader, trainDataloader *DataLoader, B, T int) error {
	fmt.Printf("train dataset num_batches: %d\n", valDataloader.numBatches)
	const genMaxLength int = 64
	genTokens := make([]int32, genMaxLength)
	valNumBatches := 10
	for step := 0; step <= 10; step++ {
		if step%10 == 0 {
			var valLoss float32
			valDataloader.Reset()
			for i := 0; i < valNumBatches; i++ {
				input, target, err := valDataloader.NextBatch()
				if err != nil {
					return err
				}
				model.forward(input, target, B, T)
				valLoss += model.MeanLoss
			}
			valLoss /= float32(valNumBatches)
			fmt.Printf("val loss %f\n", valLoss)
		}
		if step > 0 && step%20 == 0 {
			genTokens[0] = GPT2_EOT // the GPT-2 EOT token kicks off the generation
			for t := 1; t < genMaxLength; t++ {
				// for each t, we re-compute all activations between 0 and t
				// leaving this alone because you want separate code for inference anyway
				// the inference here is just for sanity checking purposes
				model.forward(genTokens, nil, 1, t)
				probabilities := model.Acts.Probs.index(t - 1).data
				coin := rand.Float32()
				nextToken2 := sampleMult(probabilities, coin)
				genTokens[t] = rune(nextToken2)
			}
			fmt.Print("generated: ")
			for t := 0; t < genMaxLength; t++ {
				fmt.Printf("%d ", genTokens[t])
			}
			fmt.Println()
		}
		// do a training step
		start := time.Now()
		input, targets, err := trainDataloader.NextBatch()
		if err != nil {
			return err
		}
		model.forward(input, targets, B, T)
		model.backward()
		model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step+1)
		fmt.Printf("step %d: train loss %f (took %v ms)\n", step, model.MeanLoss, time.Since(start))
	}
	return nil
}

func (model *GPT2) zeroGradient() {
	for i := range model.GradsActsMemory {
		model.GradsActsMemory[i] = 0.0
	}
	for i := range model.GradsMemory {
		model.GradsMemory[i] = 0.0
	}
}

func (model *GPT2) forward(input, target []int32, B, T int) {
	V := model.Config.VocabSize
	L := model.Config.NumLayers
	NH := model.Config.NumHeads
	C := model.Config.Channels
	if model.ActsMemory == nil {
		model.BatchSize = B
		model.SeqLen = T
		model.ActsMemory = model.Acts.init(B, C, T, L, NH, V)
		// also create memory for caching inputs and targets
		model.Inputs = make([]rune, B*T)
		model.Targets = make([]rune, B*T) // might be unused if we never have targets but it's small
	} else {
		// validate B,T is no larger than what was previously allocated
		// in principle, we could re-allocate a larger chunk of memory, for now we just error out
		if B != model.BatchSize || T != model.SeqLen {
			fmt.Printf("Error: batch size or sequence length is inadequately large\n")
			fmt.Printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model.BatchSize, model.SeqLen, B, T)
			return
		}
	}
	copy(model.Inputs, input)
	copy(model.Targets, target)
	params := model.Params
	acts := model.Acts
	// This encodes the word token embeddings with the positional embeddings
	// so that those vectors have spacial information and aren't just purely made up of the
	// token embeddings. The result of this is stored in acts.Encoded.
	// Input is a slice of ids/tokens that correspond to the vectors in WTE and their index is the "position"
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)
	encoderForward(acts.Encoded.data, input, params.WordTokEmbed.data, params.WordPosEmbed.data, B, T, C)
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)
	var residual []float32
	for l := 0; l < L; l++ {
		// residual is a connection between the last layers output, or the initial token/pos embedding (as applied above)

		if l == 0 {
			residual = acts.Encoded.data
		} else {
			residual = acts.Residual3.data[(l-1)*B*T*C:]
		}
		// Parameters
		l_ln1w := params.Layer1NormW.data[l*C:]
		l_ln1b := params.Layer1NormB.data[l*C:]
		l_qkvw := params.QueryKeyValW.data[l*3*C*C:]
		l_qkvb := params.QueryKeyValB.data[l*3*C:]
		l_attprojw := params.AttProjW.data[l*C*C:]
		l_attprojb := params.AttProjB.data[l*C:]
		l_ln2w := params.Layer2NormW.data[l*C:]
		l_ln2b := params.Layer2NormB.data[l*C:]
		l_fcw := params.FeedFwdW.data[l*4*C*C:]
		l_fcb := params.FeedFwdB.data[l*4*C:]
		l_fcprojw := params.FeedFwdProjW.data[l*C*4*C:]
		l_fcprojb := params.FeedFwdProjB.data[l*C:]
		// Activations
		l_ln1 := acts.Layer1Act.data[l*B*T*C:]
		l_ln1_mean := acts.Ln1Mean.data[l*B*T:]
		l_ln1_rstd := acts.Ln1Rstd.data[l*B*T:]
		l_qkv := acts.Qkv.data[l*B*T*3*C:]
		l_atty := acts.Atty.data[l*B*T*C:]
		l_preatt := acts.Preatt.data[l*B*NH*T*T:]
		l_att := acts.Att.data[l*B*NH*T*T:]
		l_attproj := acts.Attproj.data[l*B*T*C:]
		l_residual2 := acts.Residual2.data[l*B*T*C:]
		l_ln2 := acts.Ln2.data[l*B*T*C:]
		l_ln2_mean := acts.Ln2Mean.data[l*B*T:]
		l_ln2_rstd := acts.Ln2Rstd.data[l*B*T:]
		l_fch := acts.Fch.data[l*B*T*4*C:]
		l_fch_gelu := acts.FchGelu.data[l*B*T*4*C:]
		l_fcproj := acts.Fcproj.data[l*B*T*C:]
		l_residual3 := acts.Residual3.data[l*B*T*C:]

		// Here we normalise the layer so that the mean is 0 and the standard deviation is ~1.
		// residual contains the un-edited activations
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		layernormForward(l_ln1, l_ln1_mean, l_ln1_rstd, residual /*inp*/, l_ln1w /*weight*/, l_ln1b /*bias*/, B, T, C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		/*
					l_qkvw = weight = Query Key Val Weights (C * 3C)
					l_ln1 = inp = layer activations
					l_qkvb = bias = Query Key Val Bias
					l_qkv = out = key/query/value matrix
				Here we're matrix multiplying  l_ln1(inp)*l_qkvw(weight) + l_qkvb(bias)
				This matrix multiplication essentially gets a layer activation for the model inputs (activations) which are multiplied
				by the model weights.
			This does the input "projection" via linear transformations via the model query/key/value weights into higher dimensionality.
		*/
		matmulForward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		/*
			The attention forward pass takes these query/key/value vectors, along with the model attention weights
			The model pre-attention scores, after the forward pass, have the un-normalised attention scores
			att has the attention acores and l_atty has the attention scores + the query/key/value scores
			l_qkv has the projection of the activations into a higher dimension.
			l_preatt: has the projection qkv vectors dot product(similarity), between an input's query and another input's key.
				This basically goes like this:
				word a: has a query vector "what am i looking for"
				word b: has a query vector "what do i need"
				if they're similar, these vectors will be similar, therefore the scores will be high and be stored in l_preatt
			the v in the qkv is the original token/position embeddings which have been through a number of linear transformations at this point.
		*/
		attentionForward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)

		/*
			Here we do another matrix multiplication of attention weights and biases
			This projects the l_atty into another dimension. These will probably also get back propagated.
		*/
		matmulForward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		/*
			The residual forward simply adds the attention projection and the residual layer, which is the
			weights(or activations?) before any of the previous transformations. This allows a stronger signal and
			prevents weight dropout and i think makes back propagation more efficient.
		*/
		residualForward(l_residual2, residual, l_attproj, B*T*C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		/*
			The weights in this level are the layer 2 activations, which are multiplied with the residual through the above sections
			This is normalised and everything into layernorm2
		*/
		layernormForward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		/*
			Feedforward is just another layer of a multi layer perceptron to make the "higher level" connections.
		*/
		matmulForward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		/*
			This is an acitvation function which maps large values to close to one and smaller values to zero.
		*/
		geluForward(l_fch_gelu, l_fch, B*T*4*C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		/*
			This now squishes the last layer into a smaller dimension so it can be added to the next layer.
		*/
		matmulForward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		/*
			Now we set the next residual layer as the output of this layer. This is the l_fcproj + the current layer residual
		*/
		residualForward(l_residual3, l_residual2, l_fcproj, B*T*C)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
	}
	residual = acts.Residual3.data[(L-1)*B*T*C:]

	/*
		Now this is the last thing. We're layer norming the final layer activations so that the logits can be calculated

	*/
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)
	layernormForward(acts.Lnf.data, acts.LnfMean.data, acts.LnfRstd.data, residual, params.LayerFinNormW.data, params.LayerFinNormB.data, B, T, C)
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)
	/*
			Matrix multiplying the Word Token embedding gives us the logits.
		This is calculating a weighted sum. More likely tokens will be blown up and less likely will be zero or negative.
	*/
	matmulForward(acts.Logits.data, acts.Lnf.data, params.WordTokEmbed.data, nil, B, T, C, V)
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)
	/*
		After all of this we can softmax the logits to get probabilities over the entire vocabulary
	*/
	softmaxForward(acts.Probs.data, acts.Logits.data, B, T, V)
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)
	// also forward the cross-entropy loss function if we have the targets
	if len(target) > 0 {
		/*
			This compares the probabilities for each token and compares it to the target to calculate a loss.
		*/
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		crossEntropyForward(model.Acts.Losses.data, model.Acts.Probs.data, target, B, T, V)
		PrintArr("model->params_memory", model.ParamsMemory)
		PrintArr("model->grads_memory", model.GradsMemory)
		PrintArr("model->grads_acts_memory", model.GradsActsMemory)
		PrintArr("model->acts_memory", model.ActsMemory)
		// for convenience also evaluate the mean loss
		var meanLoss float32
		for i := range model.Acts.Losses.data {
			meanLoss += model.Acts.Losses.data[i]
		}
		meanLoss /= float32(B * T)
		model.MeanLoss = meanLoss

	} else {
		model.MeanLoss = -1.0
	}
	PrintArr("model->params_memory", model.ParamsMemory)
	PrintArr("model->grads_memory", model.GradsMemory)
	PrintArr("model->grads_acts_memory", model.GradsActsMemory)
	PrintArr("model->acts_memory", model.ActsMemory)

}
