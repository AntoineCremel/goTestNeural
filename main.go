package main

import (
	"fmt"
	"log"

	gor "gorgonia.org/gorgonia"
)

func main() {
	g := gor.NewGraph()

	var x, y, z *gor.Node
	var err error

	x = gor.NewScalar(g, gor.Float64, gor.WithName("x"))
	y = gor.NewScalar(g, gor.Float64, gor.WithName("y"))
	z, err = gor.Add(x, y)
	if err != nil {
		log.Fatal(err)
	}

	machine := gor.NewTapeMachine(g)

	gor.Let(x, 2.0)
	gor.Let(y, 2.5)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%v", z.Value())
}
