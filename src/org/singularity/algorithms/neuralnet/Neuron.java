package org.singularity.algorithms.neuralnet;

import java.util.*;

import org.singularity.algorithms.neuralnet.input.*;
import org.singularity.algorithms.neuralnet.output.*;

/**
 * Neural network neuron. A neuron is essentially one small unit of
 * intelligence within a layer in the neural network. Each neuron
 * contains its own classifier, normally some form of a Perceptron.
 */
public class Neuron {

	private static int _index = 0;
	
	private Layer layer;
	
	private List<Synapse> inputs = new ArrayList<Synapse>();
	
	private List<Synapse> outputs = new ArrayList<Synapse>();
	
	private NeuralNetworkConfiguration config;
	
	private double input = 0.0d;
	protected double output = 0.0d;
	private double error = 0.0d;
	private int index = _index++;
	
	/**
	 * Creates a new neuron within a neural network layer.
	 * @param layer The layer the neuron should be added to.
	 */
	public Neuron(NeuralNetworkConfiguration config, Layer layer) {
		this.config = config;
		this.layer = layer;
	}

	/**
	 * Connects this neuron to all neurons in another (subsequent)
	 * layer in the neural network. This essentially creates a new
	 * synapse between this neuron for each neuron in the target layer,
	 * allowing this neuron to send its output to all neurons in the
	 * target layer.
	 * @param layer Target layer to connect this neuron to.
	 */
	public void connectTo(Layer layer) {
	    for (Neuron neuron : layer.getNeurons()) {
	    	final Synapse synapse = new Synapse(this, neuron);
	    	this.addOutputSynapse(synapse);
	    	neuron.addInputSynapse(synapse);
	    }
	}

	public void addInputSynapse(Synapse synapse) {
		this.inputs.add(synapse);
	}

	public void addOutputSynapse(Synapse synapse) {
		this.outputs.add(synapse);
	}


	/**
	 * Returns all input synapses for this neuron.
	 * @return All input synapses for this neuron.
	 */
	public List<Synapse> getInputSynapses() {
	    return this.inputs;
	}

	/**
	 * Returns all output synapses for this neuron.
	 * @return All output synapses for this neuron.
	 */
	public List<Synapse> getOutputSynapses() {
	    return this.outputs;
	}

	/**
	 * Sets the error measure for this neuron.
	 * @param error Neuron's error measure.
	 */
	public void setError(double error) {
	    this.error = error;
	}

	/**
	 * Returns the neuron's error measure.
	 * @return The neuron's error measure.
	 */
	public double getError() {
	    return this.error;
	}

	/**
	 * Returns the output/transfer function for this neuron.
	 * @return the output/transfer function for this neuron.
	 */
	public OutputFunction getOutputFunction() {
	    return this.config.getOutputFunction();
	}

	/**
	 * Returns the input function for this neuron.
	 * @return the input function for this neuron.
	 */
	public InputFunction getInputFunction() {
	    return this.config.getInputFunction();
	}

	/**
	 * Sets the input value for this neuron.
	 * @param input The input value for this neuron.
	 */
	public void setInput(double input) {
	    this.input = input;
	}

	/**
	 * Returns the input value for this neuron.
	 * @return the input value for this neuron.
	 */
	public double getInput() {
	    return this.input;
	}

	/**
	 * Calculates the output value for this neuron given its current input.
	 */
	public void calculate() {
		if (this.inputs.size() > 0) {
			this.input = this.getInputFunction().getInput(this.inputs);
		}
	    this.output = this.getOutputFunction().getOutput(this.input);
	}

	/**
	 * Returns the current output value for this neuron.
	 * You should call Neuron.calculate() before calling this method.
	 * @return This neuron's current output value.
	 */
	public double getOutput() {
	    return this.output;
	}

	/**
	 * Resets the neuron to its default state.
	 */
	public void reset() {
	    input = 0;
	    output = 0;
	}

	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("    Neuron=[ID=");
		buf.append(this.index);
		buf.append(", Input=");
		buf.append(this.input);
		buf.append(", Output=");
		buf.append(this.output);
		buf.append(", Error=");
		buf.append(this.error);
		buf.append("\n");
		for (Synapse synapse : this.outputs) {
			buf.append("      Synapse=[Weight=");
			buf.append(synapse.getWeight().get());
			buf.append("]\n");
		}
		buf.append("    ]\n");
	    return buf.toString();
	}
}
