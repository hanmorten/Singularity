package org.singularity.algorithms.neuralnet.output;

/**
 * Sigmoid transfer function implementation.
 */
public class OutputFunctionSigmoid extends OutputFunction {

	/**
	 * Creates a new Sigmoid transfer function.
	 */
	public OutputFunctionSigmoid() {
		
	}
	
	/**
	 * Returns the output value of the neuron, based on the output/transfer function.
	 * @param net Net input value.
	 * @return Output value for the neuron.
	 */
	public double getOutput(double net) {
	    if (net > 100.0d)
	        return output = 1.0d;
	    else if (net < -100.0d)
	        return output = 0.0d;
	    
	    return this.output = (1.0d / (1.0d + Math.exp(-slope * net)));
	}

	/**
	 * Returns the derivative for the output function given the neuron's
	 * current output value. Note that for this method to return a proper
	 * value, the getOutput method must be called first!!!
	 * @return The derivative for the output function.
	 */
	public double getDerivative(double net) {
	    // This is the beauty of the sigmoid function - very easy to
	    // calculate the derivative!
	    return this.slope * this.output * (1.0d - this.output) + 0.1d;
	}


}
