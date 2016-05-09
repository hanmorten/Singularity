package org.singularity.algorithms.neuralnet.learning;

import java.util.*;

import org.apache.commons.math3.linear.RealVector;
import org.singularity.algorithms.neuralnet.*;
import org.singularity.algorithms.neuralnet.output.OutputFunction;

/**
 * Back-propagation learning rule for neural networks.
 */
public class LearningRuleBackPropagation extends LearningRule {

	/**
	 * Creates a new learning rule for a given neural network,
	 * using a given learning rate.
	 * @param rate Learning rate for this rule.
	 */
	public LearningRuleBackPropagation(double rate) {
		super(rate);
	}
	
	/**
	 * This method implements weight update procedure for the whole network
	 * for the specified  output error vector
	 */
	public void updateWeights(RealVector error) {
	    // Update the weights in the output neuron.
		this.updateOutputNeurons(error);
	    // Back-propagation: Update weights for hidden neurons.
	    this.updateHiddenNeurons();
	}

	/**
	 * Calculates the error for each output neuron and updates its weights
	 * accordingly.
	 * @param error Output neuron's error vector.
	 */
	private void updateOutputNeurons(RealVector error) {
	    int i = 0;
	    // Iterate over all output neurons
	    for (Neuron neuron : network.getOutputNeurons()) {
	        // If the error is zero, then we store a zero error and leave the weight as-is.
	    	if (error.getEntry(i) == 0.0d) {
	    		neuron.setError(0.0d);
	            i++;
	            continue;
	        }
	        
	        // Otherwise we calculate the error (delta) for the neuron...
	        final double input = neuron.getInput();
	        final OutputFunction outputFunction = neuron.getOutputFunction();
	        final double delta = error.getEntry(i) * outputFunction.getDerivative(input);
	        neuron.setError(delta);
	        
	        // ...and then update the weights for the neuron.
	        this.updateNeuronWeights(neuron);
	        i++;
	    }
	}

	/**
	 * Calculates the error for each hidden neuron and updates its weights
	 * accordingly (back-propagation).
	 */
	public void updateHiddenNeurons() {
	    // Iterate over all hidden layers in reverse order (back to front).
	    final List<Layer> layers = network.getLayers();
	    for (int i = layers.size() - 2; i > 0; i--) {
	        // Get the next layer and iterate over its neurons.
	        final Layer layer = layers.get(i);
	        for (Neuron neuron : layer.getNeurons()) {
	            // Calculate the hidden neuron's error (delta).
	            final double neuronError = this.calculateHiddenNeuronError(neuron);
	            neuron.setError(neuronError);
	            this.updateNeuronWeights(neuron);
	        }
	    }
	}

	/**
	 * Calculates and returns a hidden neuron's error (delta).
	 * @param neuron Neuron to calculate error for.
	 * @return Neuron's error (delta).
	 */
	public double calculateHiddenNeuronError(Neuron neuron) {
	    // Calculate the total error for the neuron.
	    double error = 0.0f;
	    for (Synapse synapse : neuron.getOutputSynapses()) {
	        error += synapse.getOutputNeuron().getError() * synapse.getWeight().get();
	    }
	    
	    final OutputFunction outputFunction = neuron.getOutputFunction();
	    return error * outputFunction.getDerivative(neuron.getInput());
	}

}
