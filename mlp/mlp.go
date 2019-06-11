package mlp

import (
	gor "gorgonia.org/gorgonia"
)

// NewMLP creates the VM of a Multi Layer Perceptron
// ie a basic feedforward Neural Network
func NewMLP(layers []int) gor.VM {
	network := gor.NewGraph()

	machine := gor.NewTapeMachine(network)
	return machine
}
