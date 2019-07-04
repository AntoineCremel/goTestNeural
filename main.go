package main

import (
	"fmt"
	"goTestNeural/mlp"
	"goTestNeural/util"
)

func main() {
	nn, err := mlp.NewMLPClassifier(4, []int{12, 3})

	if err != nil {
		fmt.Println("Error! ")
		fmt.Println(err)
	} else {
		fmt.Println(nn.Graph())
		util.WriteDotToFile(nn.Graph().ToDot())
	}
}
