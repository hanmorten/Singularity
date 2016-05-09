package org.singularity.algorithms.kernel;

import org.apache.commons.math3.linear.*;

/**
 * Implementation of the polynomial kernel.     
 */
public class PolynomialKernel extends Kernel {

	private double coefficient;
	private double bias;
	private double power;
	
	/**
	 * Creates a new polynomial kernel.
	 * @param coefficient
	 * @param bias
	 * @param power
	 */
	public PolynomialKernel(double coefficient, double bias, double power) {
		this.coefficient = coefficient;
		this.bias = bias;
		this.power = power;
	}
	
	/**
	 * Applies the kernel to two vectors.
	 * @param a Vector one.
	 * @param b Vector two.
	 * @return Output of the kernel function.
	 */
	public double calc(RealVector a, RealVector b) throws KernelException {
		try {
			final double base =this.coefficient * a.dotProduct(b) + bias; 
			if (this.power == 1.0d)
				return base;
			else if (this.power == 2.0d)
				return base * base;
			else if (this.power == 3.0d)
				return base * base * base;
			else
				return Math.pow(this.coefficient * a.dotProduct(b) + bias, this.power);
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
		buf.append("PolynomialKernel(");
		buf.append("Coefficient=");
		buf.append(this.coefficient);
		buf.append(", Bias=");
		buf.append(this.bias);
		buf.append(", Power=");
		buf.append(this.power);
		buf.append(")");
		return buf.toString();
	}

}
