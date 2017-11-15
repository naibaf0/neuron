package main

import (
	"fmt"
	"math/rand"

	"github.com/naibaf0/neuron"
)

func main() {
	RunSimpleNeuronTest()
}

func RunSimpleNeuronTest() {
	// set the random seed to 0
	rand.Seed(0)

	// create patterns to train the network
	patterns := [][][]float64{
		{{-2, -1}, {1}},
		{{3, 2}, {0}},
		{{-1, -0.5}, {1}},
		{{1, 1}, {0}},
	}

	// instantiate the Neuron
	n := &neuron.Neuron{}

	// initialize the Neuron with two inputs.
	n.Init(2)

	fmt.Println("Before Training:")
	n.Test(patterns)

	// train the network for 100 epochs with learning rate 0.6
	n.Train(patterns, 100)

	// testing the network
	fmt.Println("After Training:")
	n.Test(patterns)

	// predicting a value
	inputs := []float64{-3, 1}
	n.Update(inputs)
}
