package org.singularity.algorithms.regression;

import org.apache.commons.math3.linear.*;
import org.singularity.*;
import org.singularity.algorithms.LearningListenerStdout;
import org.singularity.algorithms.TrainingSample;
import org.singularity.algorithms.TrainingSet;

/**
 * Linear regression learning implementation using the least squares method.
 * Least squares is solved using QR decomposition:
 *   https://en.wikipedia.org/wiki/QR_decomposition
 */
public class LinearLeastSquaresRegression extends Regression {

	private double bias = 0.0d;

	private RealVector beta = new ArrayRealVector(1);
	
	public LinearLeastSquaresRegression() {
		
	}

	public void train(TrainingSet samples) throws RegressionException {
		if (this.listener != null) this.listener.trainingStart(this);

		final int rows = samples.size();
		final int cols = samples.getFeatureVectorSize();
		
		final double[] y = new double[rows];
		final double[][] x = new double[rows][cols+1];
		for (int r=0; r<rows; r++) {
			final TrainingSample sample = samples.get(r);
			final RealVector features = sample.getFeatures();
			y[r] = sample.getLabel();
			x[r][0] = 1.0d; 
			for (int c=0; c<cols; c++) {
				x[r][c+1] = features.getEntry(c);
			}
		}
		
        final RealMatrix xMatrix = new Array2DRowRealMatrix(x);
        final RealVector yVector = new ArrayRealVector(y);
        
        final QRDecomposition qr = new QRDecomposition(xMatrix);
        final RealVector beta = qr.getSolver().solve(yVector);

        this.bias = beta.getEntry(0);
        this.beta = beta.getSubVector(1, beta.getDimension() - 1);
        
		if (this.listener != null) this.listener.trainingEnd(this, samples);
	}

	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * @param input Feature vector of test sample.
	 * @return real value result
	 * @throws RegressionException on any error.
	 */
	public double test(RealVector input) throws RegressionException {
		return input.dotProduct(this.beta) + bias;
	}

	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("LinearLeastSquaresRegression(Bias=");
		buf.append(this.bias);
		buf.append(", Weights=[");
		for (int i=0; i<this.beta.getDimension(); i++) {
			if (i > 0) buf.append(",");
			buf.append(this.beta.getEntry(i));
		}
		buf.append("])");
		return buf.toString();
	}
	
	/**
	 * Returns the name of the algorithm.
	 * @return the name of the algorithm.
	 */
	public String name() {
		return "LinearLeastSquaredRegression";
	}

	public static void main(String[] args) {
		try {
			final TrainingSet input = new TrainingSet();

			input.add(new double[] { 1,1 },  1.1);
			input.add(new double[] { 1,2 },  1.2);
			input.add(new double[] { 1,3 },  1.3);
			input.add(new double[] { 1,4 },  1.4);
			input.add(new double[] { 1,5 },  1.5);

			input.add(new double[] { 2,1 },  2.1);
			input.add(new double[] { 2,2 },  2.2);
			input.add(new double[] { 2,3 },  2.3);
			input.add(new double[] { 2,4 },  2.4);

			input.add(new double[] { 3,1 },  3.1);
			input.add(new double[] { 3,2 },  3.2);
			input.add(new double[] { 3,3 },  3.3);
			input.add(new double[] { 3,4 },  3.4);

			final Regression regression = new LinearLeastSquaresRegression();
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
