package llmgo

import "math"

func Abs(x float32) float32 {
	if x > 0 {
		return x
	}
	return -x
}

func Cosh(x float32) float32 {
	return float32(math.Cosh(float64(x)))
}

func Exp(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

func Inf(sign int) float32 {
	return float32(math.Inf(sign))
}

func Log(x float32) float32 {
	return float32(math.Log(float64(x)))
}

func IsNaN(f float32) bool {
	return math.IsNaN(float64(f))
}

func Pow(x, y float32) float32 {
	return float32(math.Pow(float64(x), float64(y)))
}

func Sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func Tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}
