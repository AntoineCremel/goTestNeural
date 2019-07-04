package random

import (
	"math/rand"

	"gorgonia.org/tensor"
)

// UniformRandomTensor generates a tensor with shape dimensions
// The numbers of the tensor will be uniformly pseudo-random
func UniformRandomTensor(lowerBound, upperBound float64, dimensions ...int) *tensor.Dense {
	// Determine the factor by which we need to multiply each number
	factor := upperBound - lowerBound
	// Find the toatal size of our tensor
	totalSize := 1
	for _, dimension := range dimensions {
		totalSize *= dimension
	}

	// Now we generate the slice that will be used to fill the data in our tensor
	data := make([]float64, totalSize)
	for i := range data {
		data[i] = rand.Float64()*factor + lowerBound
	}

	// Generating the tensor
	tensor := tensor.New(tensor.WithBacking(data), tensor.WithShape(dimensions...))

	return tensor
}
