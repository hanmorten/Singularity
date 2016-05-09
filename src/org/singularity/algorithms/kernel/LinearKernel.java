package org.singularity.algorithms.kernel;

import org.apache.commons.math3.linear.*;

/**
 * Implementation of the linear kernel.     
 */
public class LinearKernel extends Kernel {

	/**
	 * Creates a new linear kernel.
	 */
	public LinearKernel() {
		
	}
	
	/**
	 * Applies the kernel to two vectors.
	 * @param a Vector one.
	 * @param b Vector two.
	 * @return Output of the kernel function.
	 */
	public double calc(RealVector a, RealVector b) throws KernelException {
		try {
			return a.dotProduct(b);
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
		return "LinearKernel";
	}

}
