package org.singularity.algorithms.kernel;

import org.apache.commons.math3.linear.*;

/**
 * Implementation of the radial basis function kernel.     
 */
public class RadialBasisFunctionKernel extends Kernel {

	/** Value for sigma parameter (for toString() only). */
	private double sigma;
	/** Multiplier for Euclidean distance. */
	private double multiplier;
	
	/**
	 * Creates a new radial basis function kernel.
	 * @param sigma
	 */
	public RadialBasisFunctionKernel(double sigma) {
		this.sigma = sigma;
		this.multiplier = -0.5d / (sigma * sigma);
	}
	
	/**
	 * Returns the squared Euclidean distance between two vectors.
	 * @param a First vector.
	 * @param b Second vector.
	 * @return squared Euclidean distance between the two vectors.
	 */
	private double eucledianDistance(RealVector a, RealVector b) {
		double out = 0.0d;
		for (int i=0; i<a.getDimension(); i++) {
			final double row = a.getEntry(i) - b.getEntry(i);
			out += row * row; 
		}
		return out;
	}

	/**
	 * Applies the kernel to two vectors.
	 * @param a Vector one.
	 * @param b Vector two.
	 * @return Output of the kernel function.
	 */
	public double calc(RealVector a, RealVector b) throws KernelException {
		return multiplier * this.eucledianDistance(a, b);
	}

	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("RadialBasisFunction(");
		buf.append("Sigma=");
		buf.append(this.sigma);
		buf.append(")");
		return buf.toString();
	}

}
