package mlp

import (
	"fmt"
	"testing"
)

func TestNewMLPClassifier(t *testing.T) {
	mlp, err := NewMLPClassifier(4, []int{12, 3})

	if err != nil {
		t.Errorf("%s", err)
	} else {
		fmt.Println(mlp.graph.String())
	}
}
