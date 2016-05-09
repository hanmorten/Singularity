package org.singularity.algorithms.classifier;

import java.io.Serializable;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.LearningListenerStdout;
import org.singularity.algorithms.TrainingSample;
import org.singularity.algorithms.TrainingSet;
import org.singularity.algorithms.kernel.*;

/**
 * Implements the Perceptron learning algorithm.     
 */
public class Perceptron extends Classifier implements Serializable {

	/** UID required for serialization. */
	private static final long serialVersionUID = 3293629989313592039L;

	/** Configuration: Maximum number of learning iterations. */
	private int iterations = 100;

	/** Configuration: Learning rate (used for dampening). */
	private double learningRate = 0.5d;

	/** Configuration: Kernel function. */
	private Kernel kernel;

	/** Output: Bias. */
	private double bias;
	
	/** Output: Weight assigned to each dimension. */
	private RealVector theta = new ArrayRealVector(new double[1]);
	
	/** "Pocket" algorithm, stored bias with best results. */
	private double cacheBias = 0.0;
	/** "Pocket" algorithm, stored weights with best results. */
	private RealVector cacheTetha = null;
	
	/** "Pocket" algorithm, error rate for current stored weights. */
	private double errors = Double.MAX_VALUE;
	
	/**
	 * Creates a new Perceptron learner.
	 * @param iterations Max number of learning iterations.
	 */
	public Perceptron(Kernel kernel, int iterations) {
		this.kernel = kernel;
		this.iterations = iterations;
	}
	
	/**
	 * Trains the classifier using a set of training samples.
	 * @param samples Training samples.
	 * @throws ClassifierException on any error.
	 */
	public void train(TrainingSet samples) throws ClassifierException {
		if (samples.size() == 0) return;
		
		if (this.listener != null) this.listener.trainingStart(this);

		try {
			this.bias = 0;
			this.theta = new ArrayRealVector(samples.getFeatureVectorSize());
			
			for (int i=0; i<iterations; i++) {
				if (this.listener != null) this.listener.trainingIteration(this, i, iterations);

				double errors = 0;
				
				// Store of current weights for "pocket" algorithm.
				final RealVector cacheTetha = this.theta.copy();
				final double cacheBias = this.bias;
				
				for (int j=0; j<samples.size(); j++) {
					final TrainingSample sample = samples.get(j);
					final RealVector features = sample.getFeatures();
					final double result = this.test(features);
					final double error = sample.getLabel() - result;
					if (error != 0.0d) {
						errors += 1;
						this.bias = this.bias + learningRate * error;
						for (int k=0; k<this.theta.getDimension(); k++) {
							this.theta.setEntry(k, this.theta.getEntry(k) + learningRate * error * features.getEntry(k));
						}
					}
				}

				// Simple pocket algorithm. In case the Perceptron doesn't
				// converge, we store the best classifier we've gotten so far
				// and return that if we don't converge.
				if (errors < this.errors) {
					this.errors = Math.abs(errors);
					this.cacheTetha = cacheTetha;
					this.cacheBias = cacheBias;
				}
				
				if (errors == 0) return;
			}
		}
		catch (Throwable e) {
			throw new ClassifierException("Error training perceptron: "+e.getMessage(), e);
		}
		finally {
			if (this.listener != null) this.listener.trainingEnd(this, samples);
		}

		// No convergence - used the cached theta vector as our weights.
		this.theta = this.cacheTetha;
		this.bias = this.cacheBias;
	}
	
	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * @param input Feature vector of test sample.
	 * @return -1 or +1.
	 * @throws ClassifierException on any error.
	 */
	public double test(RealVector input) throws ClassifierException {
		if (this.real(input) < 0.0)
			return -1.0;
		else
			return 1.0;
	}

	public double real(RealVector input) throws ClassifierException {
		try {
			return bias + this.kernel.calc(theta,input);
		}
		catch (Throwable e) {
			throw new ClassifierException("Error testing using perceptron: "+e.getMessage(), e);
		}
	}

	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("Perceptron(Bias=");
		buf.append(this.theta.getEntry(0));
		buf.append(", Theta=[");
		for (int i=1; i<this.theta.getDimension(); i++) {
			if (i > 1) buf.append(',');
			buf.append(this.theta.getEntry(i));
		}
		buf.append("], Kernel=");
		buf.append(this.kernel);
		buf.append(")");
		return buf.toString();
	}

	/**
	 * Returns the name of the algorithm.
	 * @return the name of the algorithm.
	 */
	public String name() {
		return "Perceptron";
	}

	public static void main(String[] args) {
		try {
			final TrainingSet input = new TrainingSet();
			input.add(new double[] { -1,-1 },  -1);
			input.add(new double[] { -1,-2 },  -1);
			input.add(new double[] { -2,-1 },  -1);
			input.add(new double[] { -2,-2 },  -1);

			input.add(new double[] { 1,1 },  -1);
			input.add(new double[] { 1,2 },  -1);
			input.add(new double[] { 2,1 },  -1);
			input.add(new double[] { 2,2 },  -1);

			input.add(new double[] { 6,6 }, 1);
			input.add(new double[] { 6,7 }, 1);
			input.add(new double[] { 7,6 }, 1);
			input.add(new double[] { 7,7 }, 1);

			input.add(new double[] { -6,-6 }, 1);
			input.add(new double[] { -6,-7 }, 1);
			input.add(new double[] { -7,-6 }, 1);
			input.add(new double[] { -7,-7 }, 1);

			final Classifier classifier = new Perceptron(new PolynomialKernel(0.3d, 0.1d, 2.0d), 150);
			//final Classifier classifier = new Perceptron(new RadialBasisFunctionKernel(1.0d), 50);
			//final Classifier classifier = new Perceptron(new SigmoidKernel(0.5, 0.2d), 50);
			//final Classifier classifier = new Perceptron(new LinearKernel(), 500);
			classifier.setLearningListener(new LearningListenerStdout());
			classifier.train(input);
			System.err.println("Accuracy: "+classifier.accuracy(input));
			
			classifier.plot2D();

			System.err.println("Predicted value for x=[1,1] is "+classifier.test(new ArrayRealVector(new double[] { 1,1 })));
			System.err.println("Predicted value for x=[1,2] is "+classifier.test(new ArrayRealVector(new double[] { 1,2 })));
			System.err.println("Predicted value for x=[2,2] is "+classifier.test(new ArrayRealVector(new double[] { 2,2 })));
			System.err.println("Predicted value for x=[2,3] is "+classifier.test(new ArrayRealVector(new double[] { 2,3 })));
			System.err.println("Predicted value for x=[2,4] is "+classifier.test(new ArrayRealVector(new double[] { 2,4 })));
			System.err.println("Predicted value for x=[2,5] is "+classifier.test(new ArrayRealVector(new double[] { 2,5 })));
			System.err.println("Predicted value for x=[2,6] is "+classifier.test(new ArrayRealVector(new double[] { 2,6 })));
			System.err.println("Predicted value for x=[6,1] is "+classifier.test(new ArrayRealVector(new double[] { 6,1 })));
			System.err.println("Predicted value for x=[6,2] is "+classifier.test(new ArrayRealVector(new double[] { 6,2 })));
			System.err.println("Predicted value for x=[6,5] is "+classifier.test(new ArrayRealVector(new double[] { 6,5 })));
			System.err.println("Predicted value for x=[6,6] is "+classifier.test(new ArrayRealVector(new double[] { 6,6 })));
			System.err.println("Predicted value for x=[6,7] is "+classifier.test(new ArrayRealVector(new double[] { 6,7 })));
			System.err.println("Predicted value for x=[6,8] is "+classifier.test(new ArrayRealVector(new double[] { 6,8 })));
		}
		catch (Throwable e) {
			System.err.println("Error: "+e.getMessage());
			e.printStackTrace(System.err);
		}

	}

}
