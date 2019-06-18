package main

import (
	"fmt"
	"testNeural/mlp"
)

func main() {
	mlp, err := mlp.NewMLPClassifier(4, []int{12, 3})

	if err != nil {
		fmt.Println("Error! ")
		fmt.Println(err)
	} else {
		fmt.Println(mlp.String())
	}
}
