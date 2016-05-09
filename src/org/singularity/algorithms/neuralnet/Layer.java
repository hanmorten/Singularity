package org.singularity.algorithms.neuralnet;

import java.util.*;

public class Layer {

	private NeuralNetworkConfiguration config;
	
	private List<Neuron> neurons = new ArrayList<Neuron>();
	
	public Layer(NeuralNetworkConfiguration config, int count, boolean bias) {
		this.config = config;
		
		for (int i=0; i<count; i++) {
			this.neurons.add(new Neuron(this.config, this));
		}

		if (bias) this.neurons.add(new BiasNeuron(this.config, this));
	}

	public void addBiasNeuron() {
		this.neurons.add(new BiasNeuron(this.config, this));
	}

	public List<Neuron> getNeurons() {
		return this.neurons;
	}	

	public void connectTo(Layer layer) {
		for (Neuron neuron : neurons) {
			neuron.connectTo(layer);
		}
	}

	public void calculate() {
		for (Neuron neuron : neurons) {
			neuron.calculate();
		}
	}

	public void reset() {
		for (Neuron neuron : neurons) {
			neuron.reset();
		}
	}

}
