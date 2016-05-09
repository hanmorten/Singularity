package org.singularity.algorithms.classifier;

import java.io.Serializable;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.LearningListenerStdout;
import org.singularity.algorithms.TrainingSample;
import org.singularity.algorithms.TrainingSet;
import org.singularity.algorithms.kernel.*;

/**
 * Implementation of Average Perceptron learning algorithm.     
 */
public class AveragePerceptron extends Classifier implements Serializable {

	/** UID required for serialization. */
	private static final long serialVersionUID = -3051559484297742984L;

	/** Configuration: Maximum number of learning iterations. */
	private int iterations = 100;

	/** Configuration: Learning rate (used for dampening). */
	private double learningRate = 0.1d;

	/** Configuration: Kernel function to use for mapping vectors. */
	private Kernel kernel;
	
	/** Output: Bias. */
	private double bias = 0.0d;
	
	/** Output: Weight assigned to each dimension. */
	private RealVector theta = new ArrayRealVector(new double[0]);

	/**
	 * Creates a new average perceptron learner.
	 * @param iterations Maximum number of learning iterations. 
	 */
	public AveragePerceptron(Kernel kernel, int iterations) {
		this.kernel = kernel;
		this.iterations = iterations;
	}

	/**
	 * Creates a new average perceptron learner (100 iterations).
	 */
	public AveragePerceptron(Kernel kernel) {
		this(kernel, 100);
	}

	/**
	 * Trains the classifier using a set of training samples.
	 * @param samples Training samples.
	 * @throws ClassifierException on any error.
	 */
	public void train(TrainingSet samples) throws ClassifierException {
		if (this.listener != null) this.listener.trainingStart(this);

		try {
			this.theta = new ArrayRealVector(samples.getFeatureVectorSize());

			final RealVector cache = new ArrayRealVector(samples.getFeatureVectorSize());

			double cacheBias = 0.0d;

			double count = 1.0d;

			for (int i=0; i<iterations; i++) {
				if (this.listener != null) this.listener.trainingIteration(this, i, iterations);
				
				int errors = 0;
				for (int j=0; j<samples.size(); j++) {
					final TrainingSample sample = samples.get(j);
					final RealVector features = sample.getFeatures();
					// Compare the test value against the actual output for this sample
					final double result = this.test(features); 
					// Update theta if there is a training error.
					final double error = sample.getLabel() - result;
					if (error != 0.0d) {
						errors += 1;
						bias = bias + learningRate * error;
						cacheBias = cacheBias + learningRate * error * count;
						for (int k=0; k<theta.getDimension(); k++) {
							// Update the entries in the theta vector
							theta.setEntry(k, theta.getEntry(k) + learningRate * error * sample.getFeatures().getEntry(k));
							// Update the entries in the cache vector 
							cache.setEntry(k, cache.getEntry(k) + count * learningRate * error * sample.getFeatures().getEntry(k));
						}
					}
					count += 1.0d;
				}
				if (errors == 0) break;
			}

			bias = bias - (cacheBias / count);
			for (int k=0; k<theta.getDimension(); k++) {
				theta.setEntry(k, theta.getEntry(k) - cache.getEntry(k) / count);
			}
		}
		catch (Throwable e) {
			throw new ClassifierException("Error training average perceptron: "+e.getMessage(), e);
		}
		finally {
			if (this.listener != null) this.listener.trainingEnd(this, samples);
		}
	}

	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * @param input Feature vector of test sample.
	 * @return -1 or +1.
	 * @throws ClassifierException on any error.
	 */
	public double test(RealVector input) throws ClassifierException {
		if (this.real(input) < 0.0d)
			return -1;
		else
			return 1;
	}

	public double real(RealVector input) throws ClassifierException {
		try {
			return bias + this.kernel.kernel(theta, input);
		}
		catch (Throwable e) {
			throw new ClassifierException("Error testing using average perceptron: "+e.getMessage(), e);
		}
	}

	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("AveragePerceptron(Bias=");
		buf.append(this.bias);
		buf.append(", Theta=(");
		for (int i=0; i<this.theta.getDimension(); i++) {
			if (i > 0) buf.append(',');
			buf.append('w');
			buf.append(i);
			buf.append('=');
			buf.append(this.theta.getEntry(i));
		}
		buf.append("), Kernel=");
		buf.append(this.kernel);
		buf.append(")");
		return buf.toString();
	}
	
	/**
	 * Returns the name of the algorithm.
	 * @return the name of the algorithm.
	 */
	public String name() {
		return "AveragePerceptron";
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

			final Classifier classifier = new AveragePerceptron(new PolynomialKernel(0.4d, 0.02d, 2.0d), 150);
			//final Classifier classifier = new AveragePerceptron(new PolynomialKernel(0.3d, 0.1d, 8d), 50);
			//final Classifier classifier = new AveragePerceptron(new LinearKernel(), 50);
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
			System.err.println("Predicted value for x=[3,3] is "+classifier.test(new ArrayRealVector(new double[] { 3,3 })));
			System.err.println("Predicted value for x=[4,4] is "+classifier.test(new ArrayRealVector(new double[] { 4,4 })));
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
