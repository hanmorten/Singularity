package org.singularity.algorithms.kernel;

import org.apache.commons.math3.linear.*;

/**
 * Implementation of the sigmoid kernel.     
 */
public class SigmoidKernel extends Kernel {

	private double coefficient;
	private double bias;
	
	/**
	 * Creates a new sigmoid kernel.
	 * @param coefficient
	 * @param bias
	 */
	public SigmoidKernel(double coefficient, double bias) {
		this.coefficient = coefficient;
		this.bias = bias;
	}
	
	/**
	 * Applies the kernel to two vectors.
	 * @param a Vector one.
	 * @param b Vector two.
	 * @return Output of the kernel function.
	 */
	public double calc(RealVector a, RealVector b) throws KernelException {
		try {
			return Math.tanh(this.coefficient * a.dotProduct(b) + this.bias);
		}
		catch (Throwable e) {
			throw new KernelException("Error calculating dot product: "+e.getMessage(), e);
		}
	}

	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("SigmoidKernel(");
		buf.append("Coefficient=");
		buf.append(this.coefficient);
		buf.append(", Bias=");
		buf.append(this.bias);
		buf.append(")");
		return buf.toString();
	}

}
