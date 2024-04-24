package llmgo

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"time"
)

const GPT2_EOT int32 = 50256

type GPT2Config struct {
	MaxSeqLen int `json:"max_seq_len"`
	V         int `json:"vocab_size"`
	L         int `json:"num_layers"`
	NH        int `json:"num_heads"`
	C         int `json:"channels"`
	EOT       int32
}

type GPT2 struct {
	Tokenizer Tokenizer
	Config    GPT2Config // Hyper-parameters of the model
	// Params has the actual weights of the model. Params.Memory is for convenience to be able to set/reset parameters simply
	Params ParameterTensors // Weights of the model
	// Grads contains the delta/gradient that will eventually be applied to the params in the model
	Grads ParameterTensors // Gradients of the weights
	// Fields for AdamW optimizer
	MMemory []float32         // First moment estimates (for AdamW)
	VMemory []float32         // Second moment estimates (for AdamW)
	Acts    ActivationTensors // Activations of the model
	// gradients of the activations
	GradsActs ActivationTensors
	B         int     // Current batch size (B)
	T         int     // Current sequence length (T)
	Inputs    []int32 // Input tokens
	Targets   []int32 // Target tokens
	MeanLoss  float32 // Mean loss after a forward pass
	Rand      *rand.Rand
}

// LoadGPT2Model loads the GPT-2 model from a checkpoint file.
func LoadGPT2Model(checkpointPath, tokenizerFile string) (*GPT2, error) {
	// File Reading
	f, err := Open(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("Error opening model file: %v", err)
	}
	defer f.Close()
	// Read Model Header
	model, err := loadFromReader(f)
	if err != nil {
		return nil, err
	}
	if tokenizerFile == "" {
		return model, err
	}
	tok, err := NewTokenizer(tokenizerFile)
	if err != nil {
		return nil, err
	}
	model.Tokenizer = tok
	return model, nil
}

func newGPT2(MaxSeqLen, V, L, NH, C int, vocab []string) GPT2 {
	model := GPT2{
		Config: GPT2Config{
			MaxSeqLen: MaxSeqLen,
			V:         V,
			L:         L,
			NH:        NH,
			C:         C,
		},
		Params:    newParameterTensors(V, C, MaxSeqLen, L),
		Tokenizer: newTokenizer(vocab),
		Rand:      rand.New(rand.NewSource(21)),
	}
	return model
}

func loadFromReader(f io.Reader) (*GPT2, error) {
	header := make([]int32, 256)
	err := binary.Read(f, binary.LittleEndian, header)
	if err != nil {
		return nil, fmt.Errorf("error reading model header: %v", err)
	}
	if header[0] != 20240326 || header[1] != 1 {
		return nil, fmt.Errorf("bad model file format")
	}
	model := &GPT2{
		Config: GPT2Config{
			MaxSeqLen: int(header[2]),
			V:         int(header[3]),
			L:         int(header[4]),
			NH:        int(header[5]),
			C:         int(header[6]),
			EOT:       GPT2_EOT,
		},
		Rand: rand.New(rand.NewSource(21)),
	}
	model.Params.Init(model.Config.V, model.Config.C, model.Config.MaxSeqLen, model.Config.L)
	if err := binary.Read(f, binary.LittleEndian, model.Params.Memory); err != nil {
		return nil, fmt.Errorf("error reading model: %v", err)
	}
	return model, nil
}

func (model *GPT2) String() string {
	var s string
	s += "[GPT-2]\n"
	s += fmt.Sprintf("max_seq_len: %d\n", model.Config.MaxSeqLen)
	s += fmt.Sprintf("vocab_size: %d\n", model.Config.V)
	s += fmt.Sprintf("num_layers: %d\n", model.Config.L)
	s += fmt.Sprintf("num_heads: %d\n", model.Config.NH)
	s += fmt.Sprintf("channels: %d\n", model.Config.C)
	s += fmt.Sprintf("num_parameters: %d\n", len(model.Params.Memory))
	return s
}

func (model *GPT2) Forward(input, target []int32, B, T int) {
	V, L, NH, C := model.Config.V, model.Config.L, model.Config.NH, model.Config.C
	if model.Acts.Memory == nil {
		model.B, model.T = B, T
		model.Acts.Init(B, C, T, L, NH, V)
		model.Inputs = make([]int32, B*T)
		model.Targets = make([]int32, B*T)
	}
	copy(model.Inputs, input)
	copy(model.Targets, target)
	params, acts := model.Params, model.Acts
	// This encodes the word token embeddings with the positional embeddings
	// so that those vectors have spacial information and aren't just purely made up of the
	// token embeddings. The result of this is stored in acts.Encoded.
	// Input is a slice of ids/tokens that correspond to the vectors in WTE and their index is the "position"
	encoderForward(acts.Encoded.data, input, params.WordTokEmbed.data, params.WordPosEmbed.data, B, T, C)
	var residual []float32
	for l := 0; l < L; l++ {
		// residual is a connection between the last layers output, or the initial token/pos embedding (as applied above)
		if l == 0 {
			residual = acts.Encoded.data
		} else {
			residual = acts.Residual3.data[(l-1)*B*T*C:]
		}
		// Parameters
		l_ln1w := params.LayerNorm1W.data[l*C:]
		l_ln1b := params.LayerNorm1B.data[l*C:]
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
		l_ln1_mean := acts.LayerNorm1Mean.data[l*B*T:]
		l_ln1_rstd := acts.LayerNorm1Rstd.data[l*B*T:]
		l_qkv := acts.QueryKeyVal.data[l*B*T*3*C:]
		l_atty := acts.AttentionInter.data[l*B*T*C:]
		l_preatt := acts.PreAttention.data[l*B*NH*T*T:]
		l_att := acts.Attention.data[l*B*NH*T*T:]
		l_attproj := acts.AttentionProj.data[l*B*T*C:]
		l_residual2 := acts.Residual2.data[l*B*T*C:]
		l_ln2 := acts.LayerNorm2Act.data[l*B*T*C:]
		l_ln2_mean := acts.LayerNorm2Mean.data[l*B*T:]
		l_ln2_rstd := acts.LayerNorm2Rstd.data[l*B*T:]
		l_fch := acts.FeedForward.data[l*B*T*4*C:]
		l_fch_gelu := acts.FeedForwardGelu.data[l*B*T*4*C:]
		l_fcproj := acts.FeedForwardProj.data[l*B*T*C:]
		l_residual3 := acts.Residual3.data[l*B*T*C:]
		// Here we normalise the layer so that the mean is 0 and the standard deviation is ~1.
		// residual contains the un-edited activations
		layernormForward(l_ln1, l_ln1_mean, l_ln1_rstd, residual /*inp*/, l_ln1w /*weight*/, l_ln1b /*bias*/, B, T, C)
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

		/*
			Here we do another matrix multiplication of attention weights and biases
			This projects the l_atty into another dimension. These will probably also get back propagated.
		*/
		matmulForward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C)
		/*
			The residual forward simply adds the attention projection and the residual layer, which is the
			weights(or activations?) before any of the previous transformations. This allows a stronger signal and
			prevents weight dropout and i think makes back propagation more efficient.
		*/
		residualForward(l_residual2, residual, l_attproj, B*T*C)
		/*
			The weights in this level are the layer 2 activations, which are multiplied with the residual through the above sections
			This is normalised and everything into layernorm2
		*/
		layernormForward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C)
		/*
			Feedforward is just another layer of a multi layer perceptron to make the "higher level" connections.
		*/
		matmulForward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C)
		/*
			This is an acitvation function which maps large values to close to one and smaller values to zero.
		*/
		geluForward(l_fch_gelu, l_fch, B*T*4*C)
		/*
			This now squishes the last layer into a smaller dimension so it can be added to the next layer.
		*/
		matmulForward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C)
		/*
			Now we set the next residual layer as the output of this layer. This is the l_fcproj + the current layer residual
		*/
		residualForward(l_residual3, l_residual2, l_fcproj, B*T*C)
	}
	residual = acts.Residual3.data[(L-1)*B*T*C:]

	/*
		Now this is the last thing. We're layer norming the final layer activations so that the logits can be calculated

	*/
	layernormForward(acts.LayerNormFinal.data, acts.LayerNormFinalMean.data, acts.LayerNormFinalStd.data, residual, params.LayerFinNormW.data, params.LayerFinNormB.data, B, T, C)
	/*
			Matrix multiplying the Word Token embedding gives us the logits.
		This is calculating a weighted sum. More likely tokens will be blown up and less likely will be zero or negative.
	*/
	matmulForward(acts.Logits.data, acts.LayerNormFinal.data, params.WordTokEmbed.data, nil, B, T, C, V)
	/*
		After all of this we can softmax the logits to get probabilities over the entire vocabulary
	*/
	softmaxForward(acts.Probabilities.data, acts.Logits.data, B, T, V)
	// also forward the cross-entropy loss function if we have the targets
	if len(target) > 0 {
		/*
			This compares the probabilities for each token and compares it to the target to calculate a loss.
		*/
		crossEntropyForward(model.Acts.Losses.data, model.Acts.Probabilities.data, target, B, T, V)
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
}

func (model *GPT2) Backward() error {
	//// double check we forwarded previously, with targets
	if model.MeanLoss == -1.0 {
		return errors.New("error: must forward with targets before backward")
	}
	// lazily allocate the memory for gradients of the weights and activations, if needed
	// convenience shortcuts
	B, T, V, L, NH, C := model.B, model.T, model.Config.V, model.Config.L, model.Config.NH, model.Config.C
	if len(model.Grads.Memory) == 0 {
		model.Grads.Init(V, C, model.Config.MaxSeqLen, L)
		model.GradsActs.Init(B, C, T, L, NH, V)
		model.ZeroGradient()
	}
	// backward pass
	params, grads, acts, gradsActs := model.Params, model.Grads, model.Acts, model.GradsActs
	// we kick off the chain by filling in dlosses with 1.0f/(B*T), to get the mean loss
	dlossMean := 1.0 / float32(B*T)
	for i := range gradsActs.Losses.data {
		gradsActs.Losses.data[i] = dlossMean
	}
	crossentropySoftmaxBackward(gradsActs.Logits.data, gradsActs.Losses.data, acts.Probabilities.data, model.Targets, B, T, V)
	matmulBackward(gradsActs.LayerNormFinal.data, grads.WordTokEmbed.data, nil, gradsActs.Logits.data, acts.LayerNormFinal.data, params.WordTokEmbed.data, B, T, C, V)
	residual := acts.Residual3.data[(L-1)*B*T*C:]       // last layer's residual
	dresidual := gradsActs.Residual3.data[(L-1)*B*T*C:] // write to last layer's residual
	layernormBackward(dresidual, grads.LayerFinNormW.data, grads.LayerFinNormB.data, gradsActs.LayerNormFinal.data, residual, params.LayerFinNormW.data, acts.LayerNormFinalMean.data, acts.LayerNormFinalStd.data, B, T, C)
	for l := L - 1; l >= 0; l-- {
		if l == 0 {
			residual = acts.Encoded.data
			dresidual = gradsActs.Encoded.data
		} else {
			residual = acts.Residual3.data[(l-1)*B*T*C:]
			dresidual = gradsActs.Residual3.data[(l-1)*B*T*C:]
		}

		// Assuming you have a 'params' variable of your ParameterTensors type
		l_ln1w := params.LayerNorm1W.data[l*C:]
		l_qkvw := params.QueryKeyValW.data[l*3*C*C:]
		l_attprojw := params.AttProjW.data[l*C*C:]
		l_ln2w := params.Layer2NormW.data[l*C:]
		l_fcw := params.FeedFwdW.data[l*4*C*C:]
		l_fcprojw := params.FeedFwdProjW.data[l*C*4*C:]
		// Gradients of weights
		dl_ln1w := grads.LayerNorm1W.data[l*C:]
		dl_ln1b := grads.LayerNorm1B.data[l*C:]
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
		l_ln1_mean := acts.LayerNorm1Mean.data[l*B*T:]
		l_ln1_rstd := acts.LayerNorm1Rstd.data[l*B*T:]
		l_qkv := acts.QueryKeyVal.data[l*B*T*3*C:]
		l_atty := acts.AttentionInter.data[l*B*T*C:]
		l_att := acts.Attention.data[l*B*NH*T*T:]
		l_residual2 := acts.Residual2.data[l*B*T*C:]
		l_ln2 := acts.LayerNorm2Act.data[l*B*T*C:]
		l_ln2_mean := acts.LayerNorm2Mean.data[l*B*T:]
		l_ln2_rstd := acts.LayerNorm2Rstd.data[l*B*T:]
		l_fch := acts.FeedForward.data[l*B*T*4*C:]
		l_fch_gelu := acts.FeedForwardGelu.data[l*B*T*4*C:]

		dl_ln1 := gradsActs.Layer1Act.data[l*B*T*C:]
		dl_qkv := gradsActs.QueryKeyVal.data[l*B*T*3*C:]
		dl_atty := gradsActs.AttentionInter.data[l*B*T*C:]
		dl_preatt := gradsActs.PreAttention.data[l*B*NH*T*T:]
		dl_att := gradsActs.Attention.data[l*B*NH*T*T:]
		dl_attproj := gradsActs.AttentionProj.data[l*B*T*C:]
		dl_residual2 := gradsActs.Residual2.data[l*B*T*C:]
		dl_ln2 := gradsActs.LayerNorm2Act.data[l*B*T*C:]
		dl_fch := gradsActs.FeedForward.data[l*B*T*4*C:]
		dl_fch_gelu := gradsActs.FeedForwardGelu.data[l*B*T*4*C:]
		dl_fcproj := gradsActs.FeedForwardProj.data[l*B*T*C:]
		dl_residual3 := gradsActs.Residual3.data[l*B*T*C:]
		residualBackward(dl_residual2, dl_fcproj, dl_residual3, B*T*C)
		matmulBackward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C)
		geluBackward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C)
		matmulBackward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C)
		layernormBackward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C)
		residualBackward(dresidual, dl_attproj, dl_residual2, B*T*C)
		matmulBackward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C)
		attentionBackward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH)
		matmulBackward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C)
		layernormBackward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C)
	}
	// Here we want to apply our gradients to our encoded data.
	encoderBackward(grads.WordTokEmbed.data, grads.WordPosEmbed.data, gradsActs.Encoded.data, model.Inputs, B, T, C)
	return nil
}

func (model *GPT2) Update(learningRate, beta1, beta2, eps, weightDecay float32, t int) {
	// Lazy memory allocation
	if model.MMemory == nil {
		model.MMemory = make([]float32, model.Params.Len())
		model.VMemory = make([]float32, model.Params.Len())
	}
	// Parameter updates
	for i := 0; i < model.Params.Len(); i++ {
		parameter := model.Params.Memory[i]
		gradient := model.Grads.Memory[i]
		// Momentum update
		m := beta1*model.MMemory[i] + (1.0-beta1)*gradient
		// RMSprop update
		v := beta2*model.VMemory[i] + (1.0-beta2)*gradient*gradient
		// Bias correction
		mHat := m / (1.0 - Pow(beta1, float32(t)))
		vHat := v / (1.0 - Pow(beta2, float32(t)))
		// Parameter update
		model.MMemory[i] = m
		model.VMemory[i] = v
		model.Params.Memory[i] -= learningRate * (mHat/(Sqrt(vHat)+eps) + weightDecay*parameter)
	}
}

func (model *GPT2) Inference(input string, B, T int) (string, error) {
	//B, T := 1, 16
	start := time.Now()
	defer func() {
		fmt.Printf("inference time took: %v\n", time.Now().Sub(start))
	}()
	tokens, err := model.Tokenizer.Encode(input)
	if err != nil {
		return "", err
	}
	if len(tokens) < T {
		for i := len(tokens); i <= T; i++ {
			tokens = append(tokens, model.Config.EOT)
		}
	}
	fmt.Printf("input is %d tokens long\n", len(tokens))
	model.Forward(tokens, tokens[1:], B, T)
	genTokens := make([]int32, B*T)
	for i := 0; i < B*T; i++ {
		genTokens[i] = model.Config.EOT
	}
	for t := 1; t < B*T; t++ {
		fmt.Printf("generating token: %d\n", t)
		// for each t, we re-compute all activations between 0 and t
		// leaving this alone because you want separate code for inference anyway
		// the inference here is just for sanity checking purposes
		model.Forward(genTokens, nil, B, t)
		probabilities := model.Acts.Probabilities.data[(t-1)*model.Config.V:]
		coin := model.Rand.Float32()
		nextToken2 := sampleMult(probabilities, coin)
		genTokens[t] = rune(nextToken2)
	}
	if model.Tokenizer.init {
		return model.Tokenizer.Decode(genTokens)
	}
	return "", errors.New("tokenizer not initialised")
}

func (model *GPT2) Train(valDataloader, trainDataloader *DataLoader, B, T int) error {
	fmt.Printf("train dataset num_batches: %d\n", valDataloader.NumBatches)
	const genMaxLength, valNumBatches = 64, 10
	genTokens := make([]int32, B*T)
	for step := 0; step <= 40; step++ {
		if step%10 == 0 {
			var valLoss float32
			valDataloader.Reset()
			for i := 0; i < valNumBatches; i++ {
				input, target, err := valDataloader.NextBatch()
				if err != nil {
					return err
				}
				model.Forward(input, target, B, T)
				valLoss += model.MeanLoss
			}
			valLoss /= float32(valNumBatches)
			fmt.Printf("val loss %f\n", valLoss)
		}
		if step > 0 && step%20 == 0 {
			for i := 0; i < B*T; i++ {
				genTokens[i] = model.Config.EOT
			}
			for t := 1; t < len(genTokens); t++ {
				// for each t, we re-compute all activations between 0 and t
				// leaving this alone because you want separate code for inference anyway
				// the inference here is just for sanity checking purposes
				model.Forward(genTokens, nil, B, t)
				probabilities := model.Acts.Probabilities.data[(t-1)*model.Config.V:]
				coin := rand.Float32()
				nextToken2 := sampleMult(probabilities, coin)
				genTokens[t] = rune(nextToken2)
			}
			fmt.Print("generated: ")
			if model.Tokenizer.init {
				str, err := model.Tokenizer.Decode(genTokens)
				if err != nil {
					return err
				}
				fmt.Println(str)
			} else {
				fmt.Println(genTokens)
			}
			for t := 0; t < genMaxLength; t++ {
				if model.Tokenizer.init {

				} else {
					fmt.Printf("%d ", genTokens[t])
				}
			}
			fmt.Println()
		}
		// do a training step
		start := time.Now()
		input, targets, err := trainDataloader.NextBatch()
		if err != nil {
			return err
		}
		model.Forward(input, targets, B, T)
		model.ZeroGradient()
		model.Backward()
		model.Update(1e-4, 0.9, 0.999, 1e-8, 0.0, step+1)
		fmt.Printf("step %d: train loss %f (took %v ms)\n", step, model.MeanLoss, time.Since(start))
	}
	return nil
}

func (model *GPT2) ZeroGradient() {
	for i := range model.GradsActs.Memory {
		model.GradsActs.Memory[i] = 0.0
	}
	for i := range model.Grads.Memory {
		model.Grads.Memory[i] = 0.0
	}
}
