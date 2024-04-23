package llmgo

func sampleMult(probabilities []float32, coin float32) int {
	var cdf float32
	for i, prob := range probabilities {
		cdf += prob
		if coin < cdf {
			return i
		}
	}
	return len(probabilities) - 1
}
