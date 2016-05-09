package org.singularity.algorithms.neuralnet;

/**
 * This is a special-case neuron that always outputs 1. This neuron is used to
 * provide a bias/offset for the neurons in subsequent layers.
 */
public class BiasNeuron extends Neuron {

	/**
	 * Creates a new neuron within a neural network layer.
	 * @param layer The layer the neuron should be added to.
	 */
	public BiasNeuron(NeuralNetworkConfiguration config, Layer layer) {
		super(config, layer);
	}

	/**
	 * Calculates the output value for this neuron given its current input.
	 */
	public void calculate() {
		this.output = 1.0d;
	}

	/**
	 * Returns the current output value for this neuron.
	 * You should call Neuron.calculate() before calling this method.
	 * @return This neuron's current output value.
	 */
	public double getOutput() {
	    return 1.0d;
	}

}
