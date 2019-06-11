package mlp

import (
	gor "gorgonia.org/gorgonia"
)

// NewMLP creates the graph of a Multi Layer Perceptron
// ie a basic feedforward Neural Network
func NewMLP(layers []int) *gor.ExprGraph {
	network := gor.NewGraph()

	return network
}
