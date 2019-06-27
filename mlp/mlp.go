package mlp

import (
	"fmt"

	gor "gorgonia.org/gorgonia"
)

// FeedForward type describes a simple, fully connected Neural Network
type FeedForward struct {
	graph   *gor.ExprGraph
	input   *gor.Node
	output  *gor.Node
	weights gor.Nodes
	biases  gor.Nodes
}

// NewMLPClassifier creates the VM of a Multi Layer Perceptron
// ie a basic feedforward Neural Network
// - The last layer is the output layer. It should be trained and
// will output one hot vectors containing the solution. If the size is 1,
// then we consider this to be a binary classification.
//
// IMPORTANT: will panick if layers contains less than one number
func NewMLPClassifier(inputs int, layers []int) (*FeedForward, error) {
	// Check that layers is of the minimum size
	if len(layers) == 0 {
		panic("Problem in NewMLPClassifier: must give at least 1 layers")
	}
	// Create the computational graph that we will use
	g := gor.NewGraph()

	// Create the input
	current := gor.NewVector(g, gor.Float64, gor.WithShape(inputs), gor.WithName("x"))
	input := current

	// Create a slice that will hold all the matrices containing the
	// weights for each layer
	weights := make(gor.Nodes, len(layers))
	// Create a slice to hold the biases
	biases := make(gor.Nodes, len(layers))

	var err error
	for i, size := range layers {
		// For each layer:
		// 1 Multiply by a matrix of weights
		var sizePrevious int
		if i == 0 {
			sizePrevious = inputs
		} else {
			sizePrevious = layers[i-1]
		}
		shape := gor.WithShape(sizePrevious, size)
		name := gor.WithName(fmt.Sprintf("W%d", i))
		weights[i] = gor.NewMatrix(g, gor.Float64, shape, name)
		current, err = gor.Mul(weights[i], current)
		if err != nil {
			return nil, err
		}

		// Add a bias
		shape = gor.WithShape(size)
		name = gor.WithName(fmt.Sprintf("b%d", i))
		biases[i] = gor.NewVector(g, gor.Float64, shape, name)
		current, err = gor.Add(biases[i], current)
		if err != nil {
			return nil, err
		}

		// Apply the acitvation function
		if i != len(layers) {
			// Use a non leaky relu for now
			current, err = gor.LeakyRelu(current, 0.)
			if err != nil {
				return nil, err
			}
		} else {
			current, err = gor.SoftMax(current)
			if err != nil {
				return nil, err
			}
		}
	}
	output := current

	// machine := gor.NewTapeMachine(g)
	return &FeedForward{
		graph:   g,
		input:   input,
		output:  output,
		weights: weights,
		biases:  biases,
	}, nil
}

// Train will take a set of input Vector Nodes and a set of expected OneHot Nodes,
// and will train the weights and biases of the network using those data.
func (n *FeedForward) Train(inputs, expected *gor.Nodes) {
	// first, we add the Loss function at the end of the Network

}
