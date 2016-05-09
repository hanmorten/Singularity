package org.singularity.algorithms.neuralnet.error;

import org.apache.commons.math3.linear.*;

/**
 * Mean squared error function for neural networks.
 */
public class ErrorFunctionMeanSquared extends ErrorFunction {

	/**
	 * Creates a new mean squared error function.
	 */
	public ErrorFunctionMeanSquared() {
		
	}
	
	/**
	 * Calculates the error between test output and desired output.
	 * @param have Output from test using current weights.
	 * @param want Output label provided in training sample.
	 * @return Error between desired output and actual output.
	 */
	public RealVector calculate(RealVector have, RealVector want) {
	    // Create a new vector to store the result in.
		final RealVector error = new ArrayRealVector(have.getDimension());
	    // Iterate over each entry in the desired and actual result.
	    for (int i = 0; i < have.getDimension(); i++) {
	        // Get the difference between the two...
	        final double err = want.getEntry(i) - have.getEntry(i);
	        // ... and store in the output vector.
	        error.setEntry(i, err);
	        // Add the square difference to the total.
	        total += (err * err);
	    }
	    count++;
	    return error;
	}

	/**
	 * Returns the total error measure accumulates in this error function,
	 * based on previous calls to #calculate(RealVector,Realvector)
	 * @return Total error measure.
	 */
	public double getTotalError() {
		return total / ( count * count );
	}
	
}
