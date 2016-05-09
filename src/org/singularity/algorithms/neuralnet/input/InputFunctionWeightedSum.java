package org.singularity.algorithms.neuralnet.input;

import java.util.List;

import org.singularity.algorithms.neuralnet.Synapse;

/**
 * Weighted sum input function for neurons. This input function simply
 * multiplies the values provided from each input synaps with the weight
 * for the synapse, and adds all these products together.
 */
public class InputFunctionWeightedSum extends InputFunction {

	/**
	 * Creates a new weighted-sum input function.
	 */
	public InputFunctionWeightedSum() {
		
	}

	/**
	 * Calculates a neuron's combined input value based on the input values
	 * provided via its input synapses.
	 * @param synapses Array of input synapses for the neuron.
	 * @return combined input value for the neuron.
	 */
	public double getInput(List<Synapse> synapses) {
	    double input = 0.0d;
	    
	    for (Synapse synapse : synapses) {
	    	input += synapse.getWeightedInput();
	    }
	    
	    return input;
	}

}
