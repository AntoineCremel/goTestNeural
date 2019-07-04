package mlp

import (
	"errors"
	"fmt"
	"log"
	"os"

	gor "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"goTestNeural/random"
)

// FeedForward type describes a simple, fully connected Neural Network
type FeedForward struct {
	// The machine on which the graph runs
	gor.VM

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
		value := gor.WithValue(random.UniformRandomTensor(0, 0.1, sizePrevious, size))
		weights[i] = gor.NewMatrix(g, gor.Float64, shape, name, value)
		current, err = gor.Mul(current, weights[i])
		if err != nil {
			return nil, err
		}

		// Add a bias
		shape = gor.WithShape(size)
		name = gor.WithName(fmt.Sprintf("b%d", i))
		zeros := tensor.Ones(tensor.Float64, size)
		zeros.AddScalar(-1., true)

		value = gor.WithValue(zeros)
		biases[i] = gor.NewVector(g, gor.Float64, shape, name, value)
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
	logger := log.New(os.Stdout, "", log.Flags())
	machine := gor.NewTapeMachine(g, gor.WithLogger(logger))
	return &FeedForward{
		VM:      machine,
		graph:   g,
		input:   input,
		output:  output,
		weights: weights,
		biases:  biases,
	}, nil
}

func (n *FeedForward) Graph() *gor.ExprGraph {
	return n.graph
}

// Activate will take input, feed it to the network, and compute the
// result of the computation.
func (n *FeedForward) Activate(input tensor.Tensor) (int, error) {
	// Check the input is of the right shape
	if !input.Shape().Eq(n.input.Shape()) {
		panic("The input vector is not of the shape expectd for this neural network.")
	}

	// Now make the input of our network equal to our own input
	gor.Let(n.input, input)

	err := n.RunAll()
	if err != nil {
		return 0, err
	}

	// After running the machine, the result is contained in output
	result := n.output.Value()

	// Result should be castable into a Tensor
	resultTensor, okTensor := result.(*tensor.Dense)
	if !okTensor {
		return 0, errors.New("The result is not a tensor like we expected")
	}

	// Get the argmax. This is our result now
	var axis int
	shape := resultTensor.Shape()
	if shape.IsRowVec() {
		axis = 2
	} else if shape.IsColVec() {
		axis = 1
	} else if len(shape) == 1 {
		axis = 1
	} else {
		panic("Why is this not a shape")
	}
	fmt.Printf("%T\n", result)
	fmt.Println(resultTensor)
	resultScalar, err := resultTensor.Argmax(axis)
	if err != nil {
		return 0, err
	}
	if !resultScalar.IsScalar() {
		panic("At the disco")
	}

	ret := resultScalar.ScalarValue().(int)

	return ret, err
}

// Train will take a set of input Vector Nodes and a set of expected OneHot Nodes,
// and will train the weights and biases of the network using those data.
func (n *FeedForward) Train(inputs, expected *gor.Nodes) {
	// first, we add the Loss function at the end of the Network

}
