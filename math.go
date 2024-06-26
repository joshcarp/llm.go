package llmgo

import (
	"math"
	"sync"
)

// encoderForward iterates through the batch/sequence and combines the word token embeddings
// with the word position embeddings. This allows out vector to encode tokens and positions in one.
func encoderForward(out []float32, inp []int32, wte []float32, wpe []float32, B, T, C int) {
	// Iterate over each batch
	for b := 0; b < B; b++ {
		// Iterate over each time step in the sequence
		for t := 0; t < T; t++ {
			// Calculate the index in the output slice. Each vector is C elements long.
			startOutIndex := b*T*C + t*C
			// Calculate the token ID index in the input
			// inp is the tokenized input, each number in inp char is an index within wte (word token embeddings)
			ix := inp[b*T+t]
			// Calculate the index in the token embeddings slice
			// inp -> id -> wte[id]
			startWteIndex := ix * int32(C)
			// Calculate the index in the position embeddings slice
			// Wpe starts at 0 (when t is zero) which is basically mapping directly to index
			startWpeIndex := t * C
			// Add the vectors from `wte` and `wpe` and store the result in `out`
			// here we combine the vectors in the C dimensions.
			for i := 0; i < C; i++ {
				out[startOutIndex+i] = wte[startWteIndex+int32(i)] + wpe[startWpeIndex+i]
			}
		}
	}
}

// encoderBackward calculates gradients during backpropagation
// Parameters:
//   - dwte: gradients with respect to word embeddings (wte)
//   - dwpe: gradients with respect to positional embeddings (wpe)
//   - dout: the gradient to apply to dwte and dwpe
//   - inp: input tokens (ids that refer to indexes within wte)
//   - B: batch size
//   - T: sequence length (number of time steps)
//   - C: embedding dimension (number of features)
func encoderBackward(dwte, dwpe []float32, dout []float32, inp []int32, B, T, C int) {
	// Iterate over the batch and time steps
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			// Calculate offsets for indexing
			doutBTOffset := b*T*C + t*C
			ix := inp[b*T+t]              // Get the input token id
			dwteIxOffset := ix * int32(C) // Calculate the offset for dwte
			dwpeTOffset := t * C          // Calculate the offset for dwpe

			// Iterate over the embedding dimension and apply computations
			for i := 0; i < C; i++ {
				// Get the gradient value from dout
				d := dout[doutBTOffset+i]
				// Update the gradients for word embeddings (dwte) and positional embeddings (dwpe)
				dwte[dwteIxOffset+int32(i)] += d
				dwpe[dwpeTOffset+i] += d
			}
		}
	}
}

// layernormForward normalizes the activations in each layer.
// It improves convergence in training and reduces sensitivity to initial parameters.
// For each vector, the mean and variance are calculated.
// Reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
// Paper: https://arxiv.org/abs/1607.06450
// Parameters:
//   - out: output activations (B,T,C)
//   - mean: mean values (B,T) for each position (b,t)
//   - rstd: reciprocal standard deviations (B,T) for each position (b,t)
//   - inp: input activations (B,T,C)
//   - weight: learnable weight (C) for scaling
//   - bias: learnable bias (C) for shifting
//   - B: batch size
//   - T: sequence length (number of time steps)
//   - C: embedding dimension (number of features)
func layernormForward(out, mean, rstd, inp, weight, bias []float32, B, T, C int) {
	var eps float32 = 1e-5
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			x := inp[b*T*C+t*C:]
			// Calculate mean
			var m float32 = 0.0
			for i := 0; i < C; i++ {
				m += x[i]
			}
			m /= float32(C)
			// Calculate variance
			var v float32 = 0.0
			for i := 0; i < C; i++ {
				xshift := x[i] - m
				v += xshift * xshift
			}
			v /= float32(C)
			// Calculate rstd (reciprocal standard deviation)
			s := 1.0 / Sqrt((v)+eps)
			// Normalize, scale, shift, and store output
			outBT := out[b*T*C+t*C:]
			for i := 0; i < C; i++ {
				// subtract mean to center data
				// divide by std to scale variance
				// (val - mean) / std
				n := s * (x[i] - m)
				// Multiply the weight
				o := n*weight[i] + bias[i]
				outBT[i] = o
			}
			// Store mean and rstd for backward pass
			mean[b*T+t] = m
			rstd[b*T+t] = s
		}
	}
}

func layernormBackward(dinp, dweight, dbias, dout, inp, weight, mean, rstd []float32, B, T, C int) {
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			baseIndex := b*T*C + t*C
			doutBT := dout[baseIndex : baseIndex+C]
			inpBT := inp[baseIndex : baseIndex+C]
			dinpBT := dinp[baseIndex : baseIndex+C]
			meanBT := mean[b*T+t]
			rstdBT := rstd[b*T+t]

			// Reduce operations
			var dnormMean float32 = 0.0
			var dnormNormMean float32 = 0.0
			for i := 0; i < C; i++ {
				normBTI := (inpBT[i] - meanBT) * rstdBT
				dnormI := weight[i] * doutBT[i]
				dnormMean += dnormI
				dnormNormMean += dnormI * normBTI
			}
			dnormMean /= float32(C)
			dnormNormMean /= float32(C)

			// Accumulation loop
			for i := 0; i < C; i++ {
				normBTI := (inpBT[i] - meanBT) * rstdBT
				dnormI := weight[i] * doutBT[i]
				dbias[i] += doutBT[i]
				dweight[i] += normBTI * doutBT[i]

				var dval float32
				dval += dnormI                  // Term 1
				dval -= dnormMean               // Term 2
				dval -= normBTI * dnormNormMean // Term 3
				dval *= rstdBT                  // Final scale
				dinpBT[i] += dval
			}
		}
	}
}

// matmulForward performs matrix multiplication and adds bias.
// Parameters:
//   - out: output matrix
//   - inp: input matrix
//   - weight: weight matrix
//   - bias: bias vector
//   - B: batch size
//   - T: sequence length (number of time steps)
//   - C: input dimension (number of features)
//   - OC: number of output channels
func matmulForward(out, inp, weight, bias []float32, B, T, C, OC int) {
	// Iterate over each batch
	var wg sync.WaitGroup
	for b := 0; b < B; b++ {
		// Iterate over each time step in the sequence
		for t := 0; t < T; t++ {
			wg.Add(1)
			go func(b, t int) {
				defer wg.Done()
				// Calculate the index in the output slice
				inp_bt := inp[b*T*C+t*C:]
				out_bt := out[b*T*OC+t*OC:]
				for o := 0; o < OC; o++ {
					var val float32
					if bias != nil {
						val = bias[o]
					}
					// Calculate the index in the weight slice
					wrow := weight[o*C:]
					// Perform the dot product between the input and weight row
					for i := 0; i < C; i++ {
						val += inp_bt[i] * wrow[i]
					}
					// Store the output value in the output slice
					out_bt[o] = val
				}
			}(b, t)
		}
	}
	wg.Wait()
}

func matmulBackward(dinp, dweight, dbias, dout, inp, weight []float32, B, T, C, OC int) {
	var wg sync.WaitGroup
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			wg.Add(1)
			go func(b, t int) {
				defer wg.Done()
				doutBt := dout[b*T*OC+t*OC:]
				dinpBt := dinp[b*T*C+t*C:]
				for o := 0; o < OC; o++ {
					wrow := weight[o*C:]
					d := doutBt[o]
					for i := 0; i < C; i++ {
						dinpBt[i] += wrow[i] * d
					}
				}
			}(b, t)
		}
	}
	wg.Wait()
	for o := 0; o < OC; o++ {
		wg.Add(1)
		go func(o int) {
			defer wg.Done()
			for b := 0; b < B; b++ {
				for t := 0; t < T; t++ {
					doutBt := dout[b*T*OC+t*OC:]
					inpBt := inp[b*T*C+t*C:]
					dwrow := dweight[o*C:]
					d := doutBt[o]
					if dbias != nil {
						dbias[o] += d
					}
					for i := 0; i < C; i++ {
						dwrow[i] += inpBt[i] * d
					}
				}
			}
		}(o)
	}
	wg.Wait()
}

// attentionForward performs the attention forward pass.
//
//	attention is the only layer that mixes information across time
//	every other operation is applied at every (b,t) position independently
//	(and of course, no layer mixes information across batch)
//
// Parameters:
//   - out: output matrix (B,T,C)
//   - preatt: pre-attention scores (B,NH,T,T)
//   - att: post-attention scores (B,NH,T,T)
//   - inp: input matrix (B,T,3C) holding Query, Key, Value vectors
//   - B: batch size
//   - T: sequence length (number of time steps)
//   - C: input dimension (number of features)
//   - NH: number of attention heads
func attentionForward(out, preatt, att, inp []float32, B, T, C, NH int) {
	C3 := C * 3  // This is the dimensions for the key, query and values
	hs := C / NH // head size
	scale := 1.0 / Sqrt(float32(hs))
	// Iterate over batch, sequence length, and number of heads
	var wg sync.WaitGroup
	for b := 0; b < B; b++ {
		// Sequence length
		for t := 0; t < T; t++ {
			for h := 0; h < NH; h++ {
				wg.Add(1)
				go func(b, t, h int) {
					defer wg.Done()
					// Calculate indices for query, pre-attention, and attention arrays
					// query is any particular input asking for information from other inputs
					queryT := inp[b*T*C3+t*C3+h*hs:] // inp[B][T][C3]
					preattBth := preatt[b*NH*T*T+h*T*T+t*T:]
					attBth := att[b*NH*T*T+h*T*T+t*T:]
					// Pass 1: Calculate query dot key and max value
					// The dot product is described in the paper as being better because
					// it can be optimized with matrix multiplication
					var maxval float32 = -10000.0
					// range from 0 to the current inp
					for t2 := 0; t2 <= t; t2++ {
						// Calculate key index for t2
						key_t2 := inp[b*T*C3+t2*C3+h*hs+C:] // +C because it's key
						// Compute dot product and update max value
						var val float32
						for i := 0; i < hs; i++ {
							val += queryT[i] * key_t2[i]
						}
						val *= scale
						if val > maxval {
							maxval = val
						}
						// preatt[b][h][t1][t2] == dot product (similarity) between query vector at position t1 and
						// key vector at t2.
						preattBth[t2] = val
					}
					// Pass 2: Calculate the exp and keep track of sum
					// Calculate exponential sum and update preatt and att arrays
					// maps the max value to zero,
					// and everything else negative.
					// when the exp function is called then the range of numbers will be
					// between 0 and e.
					var expsum float32
					for t2 := 0; t2 <= t; t2++ {
						expv := Exp((preattBth[t2]) - maxval)
						// expsum is a sum of all the exp'd pre_att values
						expsum += expv
						// att_bth[t2] is the exp'd preatt_bth[t2]
						attBth[t2] = expv
					}
					var expsum_inv float32
					if expsum != 0.0 {
						expsum_inv = 1.0 / expsum
					}
					// Pass 3: Normalize to get softmax
					// from 0 -> t2: att_bth[t2] = exp(preatt[t2]) / sum(exp(preatt[:]))
					// for everything else it's zero
					for t2 := 0; t2 < T; t2++ {
						if t2 <= t {
							attBth[t2] *= expsum_inv
						} else {
							// Causal attention mask (optional; used for debugging and comparison)
							attBth[t2] = 0.0
						}
					}

					// Pass 4: Accumulate weighted values into the output of attention
					// out = attention * values
					// The values in this instance are the initial token/position embeddings that have gone through many linear
					// transformations at this point.
					// This is simply applying the learned attention "weights" to the lkqv values.
					// These weights must change a whole bunch after back propagation.
					out_bth := out[b*T*C+t*C+h*hs:]
					for i := 0; i < hs; i++ {
						out_bth[i] = 0.0
					}
					for t2 := 0; t2 <= t; t2++ {
						value_t2 := inp[b*T*C3+t2*C3+h*hs+C*2:] // +C*2 because it's value
						att_btht2 := attBth[t2]
						for i := 0; i < hs; i++ {
							out_bth[i] += att_btht2 * value_t2[i]
						}
					}
				}(b, t, h)
			}
		}
	}
	wg.Wait()
}

// attentionBackward performs the backward pass for an attention mechanism
func attentionBackward(dinp, dpreatt, datt, dout, inp, att []float32, B, T, C, NH int) {
	// C3 is 3 times C, representing the size of Q, K, and V combined
	C3 := C * 3
	// hs is the size of each head
	hs := C / NH
	// scale is the factor used in the forward pass to scale the dot product
	scale := 1.0 / Sqrt(float32(hs))
	// Iterate through batch, time, and heads
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for h := 0; h < NH; h++ {
				// Calculate the indices for the arrays in this specific iteration
				attBTH := att[b*NH*T*T+h*T*T+t*T:]
				dattBTH := datt[b*NH*T*T+h*T*T+t*T:]
				dpreattBTH := dpreatt[b*NH*T*T+h*T*T+t*T:]
				dqueryT := dinp[b*T*C3+t*C3+h*hs:]
				queryT := inp[b*T*C3+t*C3+h*hs:]
				// Backward pass 4: value accumulation
				doutBTH := dout[b*T*C+t*C+h*hs:]
				for t2 := 0; t2 <= t; t2++ {
					valueT2 := inp[b*T*C3+t2*C3+h*hs+C*2:]
					dvalueT2 := dinp[b*T*C3+t2*C3+h*hs+C*2:]
					for i := 0; i < hs; i++ {
						// Compute gradients for attention and value accumulation
						dattBTH[t2] += valueT2[i] * doutBTH[i]
						dvalueT2[i] += attBTH[t2] * doutBTH[i]
					}
				}
				// Backward pass 2 & 3: softmax backward
				// Softmax does not require input (preatt) to backward
				for t2 := 0; t2 <= t; t2++ {
					for t3 := 0; t3 <= t; t3++ {
						var indicator float32
						if t2 == t3 {
							indicator = 1.0
						}
						localDerivative := attBTH[t2] * (indicator - attBTH[t3])
						dpreattBTH[t3] += localDerivative * dattBTH[t2]
					}
				}
				// Backward pass 1: query @ key matmul
				for t2 := 0; t2 <= t; t2++ {
					keyT2 := inp[b*T*C3+t2*C3+h*hs+C:]
					dkeyT2 := dinp[b*T*C3+t2*C3+h*hs+C:]
					for i := 0; i < hs; i++ {
						// Compute gradients for query and key
						dqueryT[i] += keyT2[i] * dpreattBTH[t2] * scale
						dkeyT2[i] += queryT[i] * dpreattBTH[t2] * scale
					}
				}
			}
		}
	}
}

var GELUSCALEFACTOR = Sqrt(2.0 / math.Pi)

// geluForward is the Gaussian Error Linear Units activation function.
// It leaves positive values mostly unchanged but
// maps negative value close to zero.
// Paper: https://arxiv.org/abs/1606.08415v5s
func geluForward(out, inp []float32, n int) {
	for i := 0; i < n; i++ {
		x := inp[i]
		cube := 0.044715 * x * x * x
		out[i] = 0.5 * x * (1.0 + Tanh(GELUSCALEFACTOR*(x+cube)))
	}
}

// geluBackward computes the backward pass of the GeLU non-linearity
func geluBackward(dinp, inp, dout []float32, n int) {
	for i := 0; i < n; i++ {
		x := inp[i]
		cube := 0.044715 * x * x * x
		tanhArg := GELUSCALEFACTOR * (x + cube)
		tanhOut := Tanh(tanhArg)
		coshfOut := Cosh(tanhArg)
		sechOut := 1.0 / (coshfOut * coshfOut)
		localGrad := 0.5*(1.0+tanhOut) + x*0.5*sechOut*GELUSCALEFACTOR*(1.0+3.0*0.044715*x*x)
		dinp[i] += localGrad * dout[i]
	}
}

func residualForward(out, inp1, inp2 []float32, N int) {
	for i := 0; i < N; i++ {
		out[i] = inp1[i] + inp2[i]
	}
}

func residualBackward(dinp1, dinp2, dout []float32, N int) {
	for i := 0; i < N; i++ {
		dinp1[i] += dout[i]
		dinp2[i] += dout[i]
	}
}

func softmaxForward(probs, logits []float32, B, T, V int) {
	var wg sync.WaitGroup
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			wg.Add(1)
			go func(b, t int) {
				defer wg.Done()
				baseIndex := b*T*V + t*V
				logitsBT := logits[baseIndex : baseIndex+V]
				probsBT := probs[baseIndex : baseIndex+V]
				// Numerical Stability
				var maxval float32 = -10000.0
				for i := 0; i < V; i++ {
					if logitsBT[i] > maxval {
						maxval = logitsBT[i]
					}
				}
				// Calculate exponentials and sum
				var sum float32
				for i := 0; i < V; i++ {
					probsBT[i] = Exp((logitsBT[i] - maxval))
					sum += probsBT[i] // Using float32 for potential precision gain
				}
				// Normalize
				for i := 0; i < V; i++ {
					probsBT[i] /= sum
				}
			}(b, t)
		}
	}
	wg.Wait()
}

// crossEntropyForward
func crossEntropyForward(losses []float32, probs []float32, targets []int32, B, T, V int) {
	// Iterate over each batch
	for b := 0; b < B; b++ {
		// Iterate over each time step in the sequence
		for t := 0; t < T; t++ {
			// Calculate the index in the probability slice
			startIndex := int32(b*T*V + t*V)
			// Get the correct index in the logits for the current batch and time step
			ix := targets[b*T+t]
			// Calculate the cross-entropy loss
			prob := probs[startIndex+ix]
			// Calculate the negative log of the probability for the correct target index
			losses[b*T+t] = -Log((prob))
		}
	}
}

// crossentropySoftmaxBackward calculates the cross entropy
func crossentropySoftmaxBackward(dlogits, dlosses, probs []float32, targets []int32, B, T, V int) {
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			baseIndex := b*T*V + t*V
			dlogitsBT := dlogits[baseIndex : baseIndex+V]
			probsBT := probs[baseIndex : baseIndex+V]
			dloss := dlosses[b*T+t]
			ix := targets[b*T+t]
			for i := 0; i < V; i++ {
				p := probsBT[i]
				var indicator float32
				if int32(i) == ix {
					indicator = 1.0
				} else {
					indicator = 0.0
				}
				dlogitsBT[i] += (p - indicator) * dloss
			}
		}
	}
}
