package org.singularity.algorithms.neuralnet.input;

import java.util.List;

import org.singularity.algorithms.neuralnet.Synapse;

/**
 * Base class for all input functions. Input functions control
 * how a neuron's input value is set based on the combination of the
 * inputs and values provided by its input synapses.
 */
public abstract class InputFunction {

	/** Supported input function types. */
	public static enum Type {
		WEIGHTED_SUM;
	}
	
	/**
	 * Creates a new input function.
	 * @param type Input function type.
	 * @return Input function instance.
	 */
	public static InputFunction getInputFunction(Type type) {
		switch (type) {
		case WEIGHTED_SUM:
			return new InputFunctionWeightedSum();
		default:
			return null; 
		}
	}
	
	/**
	 * Calculates a neuron's combined input value based on the input values
	 * provided via its input synapses.
	 * @param synapses Array of input synapses for the neuron.
	 * @return combined input value for the neuron.
	 */
	public abstract double getInput(List<Synapse> synapses);

}
