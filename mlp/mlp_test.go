package mlp

import (
	"fmt"
	"math/rand"
	"testing"

	"gorgonia.org/tensor"

	"goTestNeural/util"
)

func TestNewMLPClassifier(t *testing.T) {
	// Initialize the random seed
	// For now, we will use a static seed so that our result can be deterministic
	// Later on, we will switch to a time-based seed.
	rand.Seed(1)

	mlp, err := NewMLPClassifier(4, []int{12, 3})

	if err != nil {
		fmt.Printf("Error with the creation of the classifer: %s", err)
		return
	}
	fmt.Println(mlp.graph)
	// Print the graph to the target output
	err = util.WriteDotToFile(mlp.graph.ToDot())
	if err != nil {
		fmt.Println(err)
	}
	defer mlp.Close()

	// Make a test tensor
	test := tensor.New(tensor.WithBacking([]float64{1, 2, 3, 4}), tensor.WithShape(4, 1))
	res, err := mlp.Activate(test)
	fmt.Println(err)
	fmt.Println(res)
}
