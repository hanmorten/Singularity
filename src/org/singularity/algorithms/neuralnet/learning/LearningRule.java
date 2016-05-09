package org.singularity.algorithms.neuralnet.learning;

import java.util.*;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.neuralnet.*;

/**
 * Base class for all neural network learning rules.
 */
public abstract class LearningRule {

	/** Supported learning rule types. */
	public static enum Type {
		BACK_PROPAGATION
	}
	
	/**
	 * Creates a new learning rule.
	 * @param type Learning rule type.
	 * @param rate Learning rate.
	 * @return learning rule instance.
	 */
	public static LearningRule getLearningRule(Type type, double rate) {
		switch (type) {
		case BACK_PROPAGATION:
			return new LearningRuleBackPropagation(rate);
		default:
			return null;
		}
	}
	
	/** The learning rate for this rule. */
	protected double learningRate = 0.1d;
	
	/** Reference to the neural network this rule is used with. */
	protected NeuralNetwork network;
	
	/**
	 * Creates a new learning rule for a given neural network,
	 * using a given learning rate.
	 * @param rate Learning rate for this rule.
	 */
	public LearningRule(double rate) {
		this.learningRate = rate;
	}
	
	/**
	 * Sets the neural network this learning rule is used for.
	 * @param network Neural network this learning rule is used for.
	 */
	public void setNeuralNetwork(NeuralNetwork network) {
		this.network = network;
	}
	
	/**
	 * Sets the learning rate for this rule.
	 * @param rate Learning rate for this rule.
	 */
	public void setLearningRate(double rate) {
		this.learningRate = rate;
	}
	
	/**
	 * Returns the learning rate for this rule.
	 * @return the learning rate for this rule.
	 */
	public double getLearingRate() {
		return this.learningRate;
	}
	
	/**
	 * This method implements weight update procedure for the whole network
	 * for the specified  output error vector
	 */
	public abstract void updateWeights(RealVector error);

	/**
	 * This method implements weights update function for the single neuron.
	 * This is done by iterating over all of the neuron's input synapses and
	 * calculating the weight delta for each weight:
	 * <pre>
	 *      delta = learningRate * error * input
	 * </pre>
	 * where error is the difference between the desired and the actual output
	 * for the given neuron.
	 * @param neuron Neuron to update weights for.
	 */
	public void updateNeuronWeights(Neuron neuron) {
	    // Get the error(delta) for specified neuron,
	    final double neuronError = neuron.getError();
	    
	    // tanh can be used to minimise the impact of big error values,
	    //which can cause network instability
	    //neuronError = tanhf(neuronError);
	    
	    // iterate through all neuron's input connections
	    for (Synapse synapse : neuron.getInputSynapses()) {
	        // get the input from current connection
	        final double input = synapse.getInput();
	        // calculate the weight change
	        final double weightChange = learningRate * neuronError * input;
	        
	        // get the connection weight
	        final Synapse.Weight weight = synapse.getWeight();
	        weight.setChange(weightChange);
	        //weight.increment(weightChange);
	    }
	}

	/**
	 * Applies the updated weights after a complete learning iteration.
	 */
	public void applyWeights() {
		final List<Layer> layers = network.getLayers();
	    for (int i = layers.size() - 1; i > 0; i--) {
	        // Iterate neurons at each layer
	    	final Layer layer = layers.get(i);
	    	for (Neuron neuron : layer.getNeurons()) {
	            // iterate connections/weights for each neuron
	    		for (Synapse synapse : neuron.getInputSynapses()) {
	                // for each connection weight apply accumulated weight change
	                final Synapse.Weight weight = synapse.getWeight();
	                // apply delta weight which is the sum of delta weights in batch mode
	                weight.increment(weight.getChange());
	                // Reset delta weight.
	                weight.setChange(0.0d);
	            }
	        }
	    }
	}

	
}
