package org.singularity.algorithms.regression;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.LearningListenerStdout;
import org.singularity.algorithms.TrainingSample;
import org.singularity.algorithms.TrainingSet;

/**
 * Linear regression implementation. The input features are placed in a matrix,
 * and the matrix is multiplied with itself transposed. An LU (lower/upper)
 * decompisition (invented by Alan Turing) is then used to find the bias
 * and betas.
 */
public class LinearRegression extends Regression {

	/** Configuration: Regularization parameter. */
	private double regularization;

	/** Output: Bias (base value). */
	private double bias = 0.0d;
	
	/** Output: Weights. */
	private RealVector weights = new ArrayRealVector(1);

	/**
	 * Creates a new linear regression learner.
	 * @param regularization Regularization parameter.
	 */
	public LinearRegression(double regularization) {
		this.regularization = regularization;
	}

	/**
	 * Trains the regression algorithm using a set of training samples.
	 * @param samples Training samples.
	 * @throws RegressionException on any error.
	 */
	public void train(TrainingSet samples) throws RegressionException {
		if (this.listener != null) this.listener.trainingStart(this);

		final RealMatrix X = new Array2DRowRealMatrix(samples.getFeatureVectorSize() + 1, samples.size());
		final RealMatrix Xt = new Array2DRowRealMatrix(samples.size(), samples.getFeatureVectorSize() + 1);
		final RealVector y = new ArrayRealVector(samples.size());

		for (int r=0; r<samples.size(); r++) {
			final TrainingSample sample = samples.get(r);
			final double output = sample.getLabel();
			final RealVector features = sample.getFeatures();

			Xt.setEntry(r, 0, 1.0d);
			X.setEntry(0, r, 1.0d);
			
			for (int c=0; c<features.getDimension(); c++) {
				Xt.setEntry(r, c+1, features.getEntry(c));
				X.setEntry(c+1, r, features.getEntry(c));
			}
			y.setEntry(r, output);
		}

		RealMatrix lhs = X.multiply( Xt );
		if (this.regularization > 0.0) {
			for (int i = 0; i < lhs.getColumnDimension(); i++) {
				final double v = lhs.getEntry(i, i);
				lhs.setEntry(i, i, v + this.regularization);
			}
		}
		
		final RealVector rhs = new ArrayRealVector(Xt.getColumnDimension());
		for (int i=0; i<Xt.getColumnDimension(); i++) {
			double sum = 0;
			for (int j=0; j<y.getDimension(); j++) {
				sum += y.getEntry(j) * Xt.getEntry(j, i);
			}
			rhs.setEntry(i, sum);
		}

		this.weights = new LUDecomposition(lhs).getSolver().solve(rhs);
		
		this.bias = this.weights.getEntry(0);
		this.weights = this.weights.getSubVector(1, this.weights.getDimension() - 1);
		
		if (this.listener != null) this.listener.trainingEnd(this, samples);
	}

	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * @param input Feature vector of test sample.
	 * @return real value result
	 * @throws RegressionException on any error.
	 */
	public double test(RealVector input) throws RegressionException {
		return input.dotProduct(this.weights) + bias;
	}

	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("LinearRegression(Bias=");
		buf.append(this.bias);
		buf.append(", Weights=[");
		for (int i=0; i<this.weights.getDimension(); i++) {
			if (i > 0) buf.append(",");
			buf.append(this.weights.getEntry(i));
		}
		buf.append("])");
		return buf.toString();
	}
	
	/**
	 * Returns the name of the algorithm.
	 * @return the name of the algorithm.
	 */
	public String name() {
		return "LinearRegression";
	}

	public static void main(String[] args) {
		try {
			final TrainingSet input = new TrainingSet();

			input.add(new double[] { 1,1 },  1.0);
			input.add(new double[] { 1,2 },  1.2);
			input.add(new double[] { 1,3 },  1.4);
			input.add(new double[] { 1,4 },  1.6);
			input.add(new double[] { 1,5 },  1.8);

			input.add(new double[] { 2,1 },  2.0);
			input.add(new double[] { 2,2 },  2.2);
			input.add(new double[] { 2,3 },  2.4);
			input.add(new double[] { 2,4 },  2.6);

			input.add(new double[] { 3,1 },  3.0);
			input.add(new double[] { 3,2 },  3.2);
			input.add(new double[] { 3,3 },  3.4);
			input.add(new double[] { 3,4 },  3.6);

			final Regression regression = new LinearRegression(0.0);
			regression.setLearningListener(new LearningListenerStdout());
			regression.train(input);
			System.err.println("Accuracy: "+regression.accuracy(input));

			System.err.println("Predicted value for x=[1,1] is "+regression.test(new ArrayRealVector(new double[] { 1,1 })));
			System.err.println("Predicted value for x=[1,2] is "+regression.test(new ArrayRealVector(new double[] { 1,2 })));
			System.err.println("Predicted value for x=[1,3] is "+regression.test(new ArrayRealVector(new double[] { 1,3 })));
			System.err.println("Predicted value for x=[1,4] is "+regression.test(new ArrayRealVector(new double[] { 1,4 })));
			System.err.println("Predicted value for x=[1,5] is "+regression.test(new ArrayRealVector(new double[] { 1,5 })));
			System.err.println("Predicted value for x=[1,6] is "+regression.test(new ArrayRealVector(new double[] { 1,6 })));

			System.err.println("Predicted value for x=[2,1] is "+regression.test(new ArrayRealVector(new double[] { 2,1 })));
			System.err.println("Predicted value for x=[2,2] is "+regression.test(new ArrayRealVector(new double[] { 2,2 })));
			System.err.println("Predicted value for x=[2,3] is "+regression.test(new ArrayRealVector(new double[] { 2,3 })));
			System.err.println("Predicted value for x=[2,4] is "+regression.test(new ArrayRealVector(new double[] { 2,4 })));

			System.err.println("Predicted value for x=[3,1] is "+regression.test(new ArrayRealVector(new double[] { 3,1 })));
			System.err.println("Predicted value for x=[3,2] is "+regression.test(new ArrayRealVector(new double[] { 3,2 })));
			System.err.println("Predicted value for x=[3,3] is "+regression.test(new ArrayRealVector(new double[] { 3,3 })));
			System.err.println("Predicted value for x=[3,4] is "+regression.test(new ArrayRealVector(new double[] { 3,4 })));
			System.err.println("Predicted value for x=[3,5] is "+regression.test(new ArrayRealVector(new double[] { 3,5 })));
			System.err.println("Predicted value for x=[3,6] is "+regression.test(new ArrayRealVector(new double[] { 3,6 })));
			System.err.println("Predicted value for x=[3,7] is "+regression.test(new ArrayRealVector(new double[] { 3,7 })));
			System.err.println("Predicted value for x=[3,8] is "+regression.test(new ArrayRealVector(new double[] { 3,8 })));

			System.err.println("Predicted value for x=[4,1] is "+regression.test(new ArrayRealVector(new double[] { 4,1 })));
			System.err.println("Predicted value for x=[4,2] is "+regression.test(new ArrayRealVector(new double[] { 4,2 })));
			System.err.println("Predicted value for x=[4,3] is "+regression.test(new ArrayRealVector(new double[] { 4,3 })));
			System.err.println("Predicted value for x=[4,4] is "+regression.test(new ArrayRealVector(new double[] { 4,4 })));
		}
		catch (Throwable e) {
			System.err.println("Error: "+e.getMessage());
			e.printStackTrace(System.err);
		}
	}


}
