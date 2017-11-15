package neuron

import (
	"fmt"
	"log"
)

type Neuron struct {
	NInputs          int
	InputActivations []float64
	Output           float64
	Weights          []float64
	Threshold        float64
}

//Initialize the neuron
func (neuron *Neuron) Init(inputs int) {
	neuron.NInputs = inputs + 1 // +1 for bias

	neuron.InputActivations = vector(neuron.NInputs, 1.0)

	neuron.Weights = vector(neuron.NInputs, 0)

	for i := 0; i < neuron.NInputs; i++ {
		neuron.Weights[i] = random(-1, 1)
	}

	neuron.Threshold = 0
}

//The Update method activates the neuron; For an array of inputs it returns an output value between 0 and 1.
func (neuron *Neuron) Update(inputs []float64) float64 {
	if len(inputs) != neuron.NInputs-1 {
		log.Fatal("Error: wrong number of inputs")
	}

	for i := 0; i < neuron.NInputs-1; i++ {
		neuron.InputActivations[i] = inputs[i]
	}

	var sum float64

	for i := 0; i < neuron.NInputs; i++ {
		sum += neuron.InputActivations[i] * neuron.Weights[i]
	}

	if sum+neuron.Threshold > 0 {
		neuron.Output = 1
	} else {
		neuron.Output = 0
	}

	return neuron.Output
}

// CalculateNewWeights is used to train the neuron and adjust the weights accordingly to previews results and target values
func (neuron *Neuron) CalculateNewWeights(target float64) {

	delta := 0.0
	delta = (target - neuron.Output)

	for i := 0; i < neuron.NInputs; i++ {
		neuron.Weights[i] = neuron.Weights[i] + delta * neuron.InputActivations[i]
	}
}

// Train is used to train the neuron for a number of iterations
func (neuron *Neuron) Train(patterns [][][]float64, iterations int) {
	for i := 0; i < iterations; i++ {
		for _, p := range patterns {
			neuron.Update(p[0])

			neuron.CalculateNewWeights(p[1][0])
		}
	}
}

func (neuron *Neuron) Test(patterns [][][]float64) {
	for _, p := range patterns {
		fmt.Println(p[0], "->", neuron.Update(p[0]), " : ", p[1])
	}
}
