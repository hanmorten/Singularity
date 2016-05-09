package org.singularity.algorithms.neuralnet;

import java.util.*;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.*;
import org.singularity.algorithms.classifier.*;
import org.singularity.algorithms.neuralnet.learning.LearningRule;

/**
 * Neural network supervised learning algorithm implementation.
 */
public class NeuralNetwork extends Classifier {

	/** Layers in the neural network. */
	private List<Layer> layers = new ArrayList<Layer>();
	
	/** Neural network configuration. */
	private NeuralNetworkConfiguration config;

	/**
	 * Creates a new neural network.
	 */
	public NeuralNetwork(NeuralNetworkConfiguration configuration) {
		this.config = configuration;
		for (int i=0; i<this.config.getLayers(); i++) {
			this.layers.add(new Layer(this.config, this.config.getNeurons(i), this.config.useBias(i)));
		}

		Layer prev = null;
		for (Layer layer : this.layers) {
			if (prev != null) prev.connectTo(layer);
			prev = layer;
		}
		
		configuration.getLearningRule().setNeuralNetwork(this);
	}

	public NeuralNetworkConfiguration getConfiguration() {
		return this.getConfiguration();
	}
	
	public List<Layer> getLayers() {
		return this.layers;
	}

	public Layer getInputLayer() {
		return this.layers.get(0);
	}

	public List<Neuron> getInputNeurons() {
		return this.getInputLayer().getNeurons();
	}

	public Layer getOutputLayer() {
		return this.layers.get(this.layers.size() - 1);
	}

	public List<Neuron> getOutputNeurons() {
		return this.getOutputLayer().getNeurons();
	}

	public void train(TrainingSet samples) {
		if (this.listener != null) this.listener.trainingStart(this);

		for (int i=0; i<samples.size(); i++) {
			final TrainingSample sample = samples.get(i);
			if (sample.getLabel() < 0.0d) sample.setLabel(0.0d);
		}

		// Use this as workaround for single-label training samples.
		final RealVector template = new ArrayRealVector(1);

		// Main training iteration loop.
		for (int j=0; j<this.config.getIterations(); j++) {

			// Reset total error for this iteration.
			double total = 0.0d;
			this.config.getErrorFunction().reset();

			// Iterate over all training samples.
			for (int i=0; i<samples.size(); i++) {

				// Run a test using the current sample.
				final TrainingSample sample = samples.get(i);
				final RealVector input = sample.getFeatures();
				this.setInput(input);
				this.calculate();

				// Get the actual output with current network
				final RealVector have = this.getOutput();
				// Get the desired output from training sample
				RealVector want = sample.getLabels();
				// Workaround for single-label training samples.
				if (want == null) {
					want = template;
					want.setEntry(0, sample.getLabel());
				}

				// Update the network according to the training error
				final RealVector error = this.config.getErrorFunction().calculate(have, want);
				this.config.getLearningRule().updateWeights(error);
				for (int k=0; k<error.getDimension(); k++) {
					total += Math.abs(error.getEntry(k));
				}
			}

			// Apply the adjusted weights to the network.
			this.config.getLearningRule().applyWeights();

			if (listener != null) {
				listener.trainingIteration(this, j, this.config.getIterations());
			}

			// Stop iteration if we're close enough to convergence
			if (total <= this.config.getConvergenceThreshold()) {
				System.out.println("Network converged at "+j+" iterations (error = "+total+")");
				break;
			}
		}

		if (listener != null) listener.trainingEnd(this, samples);
	}

	/**
	 * Sets the input for the network.
	 */
	public void setInput(RealVector input) {
		final List<Neuron> neurons = this.getInputNeurons();
		for (int i=0; i<input.getDimension(); i++) {
			final Neuron neuron = neurons.get(i);
			neuron.setInput(input.getEntry(i));
		}
	}

	/**
	 * Performs calculation on whole network
	 */
	public void calculate() {
		for (Layer layer : this.layers) {
			layer.calculate();
		}
	}

	public void reset() {
		for (Layer layer : this.layers) {
			layer.reset();
		}
	}

	/**
	 * Returns the output for the network
	 */
	public RealVector getOutput() {
		final List<Neuron> neurons = this.getOutputNeurons();
		final int count = neurons.size();
		final RealVector output = new ArrayRealVector(count);
		for (int i = 0; i < count; i++) {
			final Neuron neuron = neurons.get(i);
			output.setEntry(i, neuron.getOutput());
		}
		return output;
	}

	public RealVector testMultiple(RealVector input) throws ClassifierException {
		this.setInput(input);
		this.calculate();
		return this.getOutput();
	}

	public double real(RealVector input) throws ClassifierException {
		return this.testMultiple(input).getEntry(0);
	}

	public double test(RealVector input) throws ClassifierException {
		if (this.real(input) < 0.5d) {
			return 0.0d;
		}
		else {
			return 1.0d;
		}
	}

	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("NeuralNetwork=[\n");
		for (Layer layer : layers) {
			buf.append("  Layer = [\n");
			for (Neuron neuron : layer.getNeurons()) {
				buf.append(neuron);
			}
			buf.append("  ]\n");
		}
		buf.append("]");
		return buf.toString();
	}

	/**
	 * Returns the name of the algorithm.
	 * @return the name of the algorithm.
	 */
	public String name() {
		return "NeuralNetwork";
	}

	/**
	 * Very basic plotting of 2-dimensional problem sets.
	 * @throws ClassifierException on testing errors.
	 */
	public void plot2D(TrainingSet set) throws ClassifierException {
		final Map<RealVector,Integer> samples = new HashMap<RealVector,Integer>();
		for (int i=0; i<set.size(); i++) {
			final TrainingSample sample = set.get(i);
			samples.put(sample.getFeatures(), new Integer((int)sample.getLabel()));
		}

		final double[] data = new double[2];
		System.out.println("    -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9 ");
		for (int x1 = 9; x1>-10; x1--) {
			System.out.print(" ");
			if (x1 >= 0) System.out.print(" ");
			System.out.print(x1);
			System.out.print(" ");
			for (int x2 = -9; x2<10; x2++) {
				data[0] = x1;
				data[1] = x2;
				final RealVector x = new ArrayRealVector(data);
				final Integer label = samples.get(x);

	            if (this.test(x) <= 0.5d) {
	                if (label == null) {
	                	System.out.print("   ");
	                }
	                else if (label <= 0.5d) {
	                	System.out.print("[ ]");
	                }
	                else {
	                	System.out.print("< >");
	                }
	            }
	            else {
	                if (label == null) {
	                	System.out.print(" # ");
	                }
	                else if (label >= 0.5d) {
	                	System.out.print("[#]");
	                }
	                else {
	                	System.out.print("<#>");
	                }
	            }
			}
			System.out.println();
		}	
	}

	public static void main(String[] args) {
		try {
			final TrainingSet input = new TrainingSet();
			
			input.add(new double[] { -8,8 },  1);
			input.add(new double[] { -7,7 },  1);

			input.add(new double[] { -1,-1 },  -1);
			input.add(new double[] { -1,-2 },  -1);
			input.add(new double[] { -2,-1 },  -1);
			input.add(new double[] { -2,-2 },  -1);

			input.add(new double[] { 1,1 },  1);
			input.add(new double[] { 1,2 },  1);
			input.add(new double[] { 2,1 },  1);
			input.add(new double[] { 2,2 },  1);

			input.add(new double[] { 6,1 }, 1);
			input.add(new double[] { 7,1 }, 1);
			input.add(new double[] { 6,2 }, 1);
			input.add(new double[] { 7,2 }, 1);

			input.add(new double[] { 6,7 },  1);
			input.add(new double[] { 7,7 },  1);
			input.add(new double[] { 6,6 },  1);
			input.add(new double[] { 7,6 },  1);
			
			input.add(new double[] { 6,7 },  1);
			input.add(new double[] { 7,7 },  1);
			input.add(new double[] { 6,6 },  1);
			input.add(new double[] { 7,6 },  1);

			input.add(new double[] { 2,4 },  1);
			input.add(new double[] { 2,6 },  1);
			input.add(new double[] { 3,4 },  1);
			input.add(new double[] { 3,6 },  1);
			
			input.add(new double[] { 4,4 },  1);
			input.add(new double[] { -4,-6 },  -1);
			input.add(new double[] { -5,-4 },  -1);
			input.add(new double[] { 0,-4 },  -1);
			input.add(new double[] { -2,-4 },  -1);
			input.add(new double[] { -1,-3 },  -1);

			input.add(new double[] { 5,6 },  1);

			input.add(new double[] { 1,-8 },  1);
			input.add(new double[] { 1,-9 },  1);
			input.add(new double[] { 0,-8 },  1);
			input.add(new double[] { 0,-9 },  1);

			input.add(new double[] { -8,-8 },  -1);
			input.add(new double[] { -8,-9 },  -1);

			final NeuralNetworkConfiguration config = new NeuralNetworkConfiguration();
			config.setIterations(5000);
			config.setLearningRule(LearningRule.Type.BACK_PROPAGATION, 0.5);
			config.addLayer(2, true);
			config.addLayer(250, true);
			//config.addLayer(100, true);
			//config.addLayer(25, true);
			config.addLayer(1, false);
			
			final NeuralNetwork classifier = new NeuralNetwork(config);
			classifier.setLearningListener(new LearningListenerStdout());

			classifier.train(input);
			classifier.plot2D(input);
			System.err.println("Accuracy: "+classifier.accuracy(input));

			/*
			classifier.plot2D(input);

			System.err.println("Predicted value for x=[1,1] is "+classifier.real(new ArrayRealVector(new double[] { 1,1 })));
			System.err.println("Predicted value for x=[1,2] is "+classifier.real(new ArrayRealVector(new double[] { 1,2 })));
			System.err.println("Predicted value for x=[2,2] is "+classifier.real(new ArrayRealVector(new double[] { 2,2 })));
			System.err.println("Predicted value for x=[2,3] is "+classifier.real(new ArrayRealVector(new double[] { 2,3 })));
			System.err.println("Predicted value for x=[2,4] is "+classifier.real(new ArrayRealVector(new double[] { 2,4 })));
			System.err.println("Predicted value for x=[2,5] is "+classifier.real(new ArrayRealVector(new double[] { 2,5 })));
			System.err.println("Predicted value for x=[2,6] is "+classifier.real(new ArrayRealVector(new double[] { 2,6 })));
			System.err.println("Predicted value for x=[3,3] is "+classifier.real(new ArrayRealVector(new double[] { 3,3 })));
			System.err.println("Predicted value for x=[4,4] is "+classifier.real(new ArrayRealVector(new double[] { 4,4 })));
			System.err.println("Predicted value for x=[6,1] is "+classifier.real(new ArrayRealVector(new double[] { 6,1 })));
			System.err.println("Predicted value for x=[6,2] is "+classifier.real(new ArrayRealVector(new double[] { 6,2 })));
			System.err.println("Predicted value for x=[6,5] is "+classifier.real(new ArrayRealVector(new double[] { 6,5 })));
			System.err.println("Predicted value for x=[6,6] is "+classifier.real(new ArrayRealVector(new double[] { 6,6 })));
			System.err.println("Predicted value for x=[6,7] is "+classifier.real(new ArrayRealVector(new double[] { 6,7 })));
			System.err.println("Predicted value for x=[6,8] is "+classifier.real(new ArrayRealVector(new double[] { 6,8 })));
			 */
		}
		catch (Throwable e) {
			System.err.println("Error: "+e.getMessage());
			e.printStackTrace(System.err);
		}

	}

}
