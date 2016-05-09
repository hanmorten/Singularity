package org.singularity.algorithms.regression;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.LearningListenerStdout;
import org.singularity.algorithms.TrainingSet;

/**
 * Logistic regression implementation. Note that logistic regression
 * outputs binary -1 or +1, as a classifier.     
 */
public class LogisticRegression extends Regression {

	/** Configuration: The learning rate */
	private double rate;

	/** Configuration: The number of iterations */
	private int iterations = 100;

	/** The weight to learn */
	private double[] weights = new double[1];

	/**
	 * Creates a new instance of the logistic regression learner.
	 * @param iterations Number of learning iterations.
	 * @param rate Learning rate.
	 */
	public LogisticRegression(int iterations, double rate) {
		this.iterations = iterations; 
		this.rate = rate;
	}

	/**
	 * Trains the regression algorithm using a set of training samples.
	 * @param samples Training samples.
	 * @throws RegressionException on any error.
	 */
	public void train(TrainingSet samples) throws RegressionException {
		if (this.listener != null) this.listener.trainingStart(this);

		try {
			if (samples.size() == 0) return;

			this.weights = new double[samples.get(0).getFeatures().getDimension()];

			for (int n=0; n<iterations; n++) {
				if (this.listener != null) this.listener.trainingIteration(this, n, iterations);

				for (int i=0; i<samples.size(); i++) {
					final RealVector x = samples.get(i).getFeatures();
					final double predicted = this.test(x);
					final double label = samples.get(i).getLabel();
					for (int j=0; j<weights.length; j++) {
						weights[j] = weights[j] + rate * (label - predicted) * x.getEntry(j);
					}
				}
			}
		}
		finally {
			if (this.listener != null) listener.trainingEnd(this, samples);
		}
	}

	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * @param input Feature vector of test sample.
	 * @return binary outcome (-1 or +1).
	 * @throws RegressionException on any error.
	 */
	public double test(RealVector input) throws RegressionException {
		double value = 0.0d;
		for (int i=0; i<weights.length;i++)  {
			value += weights[i] * input.getEntry(i);
		}
		value = 1 / (1 + Math.exp(value));
		if (value < 0.5)
			return -1.0;
		else
			
			return 1.0;
	}

	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("LogisticRegression(");
		for (int i=0; i<this.weights.length; i++) {
			if (i > 0) buf.append(",");
			buf.append(this.weights[i]);
		}
		buf.append(")");
		return buf.toString();
	}

	/**
	 * Returns the name of the algorithm.
	 * @return the name of the algorithm.
	 */
	public String name() {
		return "LogisticRegression";
	}

	public static void main(String[] args) {
		try {
			final TrainingSet input = new TrainingSet();

			input.add(new double[] { 1,-1,-1 },  1);
			input.add(new double[] { 1,-1,-2 },  1);
			input.add(new double[] { 1,-2,-1 },  1);
			input.add(new double[] { 1,-2,-2 },  1);

			input.add(new double[] { 1,1,1 },  1);
			input.add(new double[] { 1,1,2 },  1);
			input.add(new double[] { 1,2,1 },  1);
			input.add(new double[] { 1,2,2 },  1);

			input.add(new double[] { 1,6,6 }, -1);
			input.add(new double[] { 1,7,6 }, -1);
			input.add(new double[] { 1,6,7 }, -1);
			input.add(new double[] { 1,7,7 }, -1);

			final Regression regression = new LogisticRegression(3000, 0.0001);
			regression.setLearningListener(new LearningListenerStdout());
			regression.train(input);
			System.err.println("Accuracy: "+regression.accuracy(input));

			System.err.println("Predicted value for x=[1,1] is "+regression.test(new ArrayRealVector(new double[] { 1,1,1 })));
			System.err.println("Predicted value for x=[1,2] is "+regression.test(new ArrayRealVector(new double[] { 1,1,2 })));
			System.err.println("Predicted value for x=[1,3] is "+regression.test(new ArrayRealVector(new double[] { 1,1,3 })));
			System.err.println("Predicted value for x=[1,4] is "+regression.test(new ArrayRealVector(new double[] { 1,1,4 })));
			System.err.println("Predicted value for x=[1,5] is "+regression.test(new ArrayRealVector(new double[] { 1,1,5 })));
			System.err.println("Predicted value for x=[1,6] is "+regression.test(new ArrayRealVector(new double[] { 1,1,6 })));

			System.err.println("Predicted value for x=[2,1] is "+regression.test(new ArrayRealVector(new double[] { 1,2,1 })));
			System.err.println("Predicted value for x=[2,2] is "+regression.test(new ArrayRealVector(new double[] { 1,2,2 })));
			System.err.println("Predicted value for x=[2,3] is "+regression.test(new ArrayRealVector(new double[] { 1,2,3 })));
			System.err.println("Predicted value for x=[2,4] is "+regression.test(new ArrayRealVector(new double[] { 1,2,4 })));

			System.err.println("Predicted value for x=[3,1] is "+regression.test(new ArrayRealVector(new double[] { 1,3,1 })));
			System.err.println("Predicted value for x=[3,2] is "+regression.test(new ArrayRealVector(new double[] { 1,3,2 })));
			System.err.println("Predicted value for x=[3,3] is "+regression.test(new ArrayRealVector(new double[] { 1,3,3 })));
			System.err.println("Predicted value for x=[3,4] is "+regression.test(new ArrayRealVector(new double[] { 1,3,4 })));
			System.err.println("Predicted value for x=[3,5] is "+regression.test(new ArrayRealVector(new double[] { 1,3,5 })));
			System.err.println("Predicted value for x=[3,6] is "+regression.test(new ArrayRealVector(new double[] { 1,3,6 })));
			System.err.println("Predicted value for x=[3,7] is "+regression.test(new ArrayRealVector(new double[] { 1,3,7 })));
			System.err.println("Predicted value for x=[3,8] is "+regression.test(new ArrayRealVector(new double[] { 1,3,8 })));

			System.err.println("Predicted value for x=[4,1] is "+regression.test(new ArrayRealVector(new double[] { 1,4,1 })));
			System.err.println("Predicted value for x=[4,2] is "+regression.test(new ArrayRealVector(new double[] { 1,4,2 })));
			System.err.println("Predicted value for x=[4,3] is "+regression.test(new ArrayRealVector(new double[] { 1,4,3 })));
			System.err.println("Predicted value for x=[4,4] is "+regression.test(new ArrayRealVector(new double[] { 1,4,4 })));
		}
		catch (Throwable e) {
			System.err.println("Error: "+e.getMessage());
			e.printStackTrace(System.err);
		}
	}

}