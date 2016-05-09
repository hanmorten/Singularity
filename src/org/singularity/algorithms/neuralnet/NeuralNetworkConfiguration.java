package org.singularity.algorithms.neuralnet;

import org.singularity.algorithms.neuralnet.error.ErrorFunction;
import org.singularity.algorithms.neuralnet.input.InputFunction;
import org.singularity.algorithms.neuralnet.output.OutputFunction;
import org.singularity.algorithms.neuralnet.learning.LearningRule;

import java.util.*;

public class NeuralNetworkConfiguration {

	private class Layer {

		private int neurons;
		private boolean bias;

		public Layer(int neurons, boolean bias) {
			this.neurons = neurons;
			this.bias = bias;
		}
		
		public int getNeurons() {
			return this.neurons;
		}
		
		public boolean useBias() {
			return this.bias;
		}
		
	}
	
	private List<Layer> layers = new ArrayList<Layer>();
	
	private ErrorFunction errorFunction = null;
	private InputFunction inputFunction = null;
	private OutputFunction outputFunction = null;
	private LearningRule learningRule = null;
	
	private double threshold = 0.0001d;
	private int iterations = 5000;
	
	public NeuralNetworkConfiguration() {
		
	}
	
	public void setConvergenceThreshold(double threshold) {
		this.threshold = threshold;
	}
	
	public double getConvergenceThreshold() {
		return this.threshold;
	}
	
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}
	
	public int getIterations() {
		return this.iterations;
	}
	
	public void setErrorFunction(ErrorFunction.Type type) {
		this.errorFunction = ErrorFunction.getErrorFunction(type);
	}
	
	public ErrorFunction getErrorFunction() {
		if (this.errorFunction == null)
			this.errorFunction = ErrorFunction.getErrorFunction(ErrorFunction.Type.MEAN_SQUARED);
		return this.errorFunction;
	}

	public void setInputFunction(InputFunction.Type type) {
		this.inputFunction = InputFunction.getInputFunction(type);
	}
	
	public InputFunction getInputFunction() {
		if (this.inputFunction == null)
			this.inputFunction = InputFunction.getInputFunction(InputFunction.Type.WEIGHTED_SUM);
		return this.inputFunction;
	}

	public void setOutputFunction(OutputFunction.Type type) {
		this.outputFunction = OutputFunction.getOutputFunction(type);
	}
	
	public OutputFunction getOutputFunction() {
		if (this.outputFunction == null)
			this.outputFunction = OutputFunction.getOutputFunction(OutputFunction.Type.SIGMOID);
		return this.outputFunction;
	}

	public void setLearningRule(LearningRule.Type type, double rate) {
		this.learningRule = LearningRule.getLearningRule(type, rate);
	}
	
	public LearningRule getLearningRule() {
		if (this.learningRule == null)
			this.learningRule = LearningRule.getLearningRule(LearningRule.Type.BACK_PROPAGATION, 0.1d);
		return this.learningRule;
	}

	public void addLayer(int neurons, boolean bias) {
		this.layers.add(new Layer(neurons, bias));
	}
	
	public int getLayers() {
		return this.layers.size();
	}
	
	public int getNeurons(int layer) {
		if (layer < layers.size())
			return layers.get(layer).getNeurons();
		else
			return 0;
	}
	
	public boolean useBias(int layer) {
		if (layer < layers.size())
			return layers.get(layer).useBias();
		else
			return false;
	}

}
