package mlp

import (
	"fmt"
	"math/rand"
	"testing"

	"gorgonia.org/tensor"
)

func TestNewMLPClassifier(t *testing.T) {
	// Initialize the random seed
	rand.Seed(3)

	mlp, err := NewMLPClassifier(4, []int{12, 3})

	if err != nil {
		t.Errorf("%s", err)
	} else {
		fmt.Println(mlp.graph)
		// Print the graph to
	}
	defer mlp.Close()

	// Make a test tensor
	test := tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4}), tensor.WithShape(4, 1))
	res, err := mlp.Activate(test)
	fmt.Println(err)
	fmt.Println(res)
}
