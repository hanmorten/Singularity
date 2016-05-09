package org.singularity.algorithms.neuralnet.output;

/**
 * Base class for output functions (or transfer functions), used to
 * transfer output values from one neuron to another, and also to
 * calculate the derivative of the output (for backpropagation of
 * learning errors to hidden/input neurons).
 */
public abstract class OutputFunction {

	/** Supported output function types. */
	public static enum Type {
		SIGMOID
	}
	
	/**
	 * Creates a new output function.
	 * @param type Output function type.
	 * @return Output function instance.
	 */
	public static OutputFunction getOutputFunction(Type type) {
		switch (type) {
		case SIGMOID:
			return new OutputFunctionSigmoid();
		default:
			return null;
		}
	}
	
	/** Slope of the output function. */
	protected double slope = 1.0d;
	/** Output value for the function. */
	protected double output = 0.0d;
	
	/**
	 * Creates a new output function.
	 */
	protected OutputFunction() {

	}

	/**
	 * Returns the slope of the function (default is 1).
	 * @return the slope of the function (default is 1).
	 */
	public double getSlope() {
		return this.slope;
	}
	
	/**
	 * Sets the slope of the function (default is 1).
	 * @param slope The slope of the function (default is 1).
	 */
	public void setSlope(double slope) {
		this.slope = slope;
	}
	
	/**
	 * Returns the output value of the neuron, based on the output/transfer function.
	 * @param net Net input value.
	 * @return Output value for the neuron.
	 */
	public abstract double getOutput(double net);

	/**
	 * Returns the derivative for the output function given the neuron's
	 * current output value. Note that for this method to return a proper
	 * value, the getOutput method must be called first!!!
	 * @param net Net input value.
	 * @return The derivative for the output function.
	 */
	public abstract double getDerivative(double net);

}
