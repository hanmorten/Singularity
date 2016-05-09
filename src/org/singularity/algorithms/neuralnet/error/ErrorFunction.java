package org.singularity.algorithms.neuralnet.error;

import org.apache.commons.math3.linear.*;

/**
 * Base class for neural network error functions.
 */
public abstract class ErrorFunction {

	/** Supported error function types. */
	public static enum Type {
		MEAN_SQUARED
	}

	/**
	 * Creates a new error function.
	 * @param type Error function type.
	 * @return Error function instance.
	 */
	public static ErrorFunction getErrorFunction(Type type) {
		switch (type) {
		case MEAN_SQUARED:
			return new ErrorFunctionMeanSquared();
		default:
			return null;
		}
	}

	/** Total error measure. */
	protected double total = 0.0d;
	/** Number of errors. */
	protected double count = 0.0d;
	
	/**
	 * Creates a new error function.
	 */
	protected ErrorFunction() {

	}

	/**
	 * Calculates the error between test output and desired output.
	 * @param have Output from test using current weights.
	 * @param want Output label provided in training sample.
	 * @return Error between desired output and actual output.
	 */
	public abstract RealVector calculate(RealVector have, RealVector want);

	/**
	 * Returns the total error measure accumulates in this error function,
	 * based on previous calls to #calculate(RealVector,RealVector)
	 * @return Total error measure.
	 */
	public abstract double getTotalError();

	/**
	 * Resets the error function.
	 */
	public void reset() {
	    total = 0.0d;
	    count = 0.0d;
	}

}
