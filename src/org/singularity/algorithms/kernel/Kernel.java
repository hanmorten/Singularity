package org.singularity.algorithms.kernel;

import java.util.*;

import org.apache.commons.math3.linear.*;

/**
 * Base class for all kernel implementation, including a cache.     
 */
public abstract class Kernel {

	/**
	 * Cache for all kernel operations.
	 */
	private Map<RealVector,Map<RealVector,Double>> cache =
			new HashMap<RealVector,Map<RealVector,Double>>();

	/**
	 * Default constructor.
	 */
	public Kernel() {
		
	}
	
	/**
	 * Applies the kernel to two vectors. The kernel function output is
	 * cached for performance, as the kernel function is generally the most
	 * frequently applied code in any learning algorithm.
	 * @param a Vector one.
	 * @param b Vector two.
	 * @return Output of the kernel function.
	 */
	public double kernel(RealVector a, RealVector b) throws KernelException {
		if (1 == 1) return this.calc(a, b);
		
		Map<RealVector,Double> map = this.cache.get(a);
		if (map == null) {
			RealVector c = a;
			a = b;
			b = c;
			map = this.cache.get(a);
		}
		if (map == null) {
			this.cache.put(a, map = new HashMap<RealVector,Double>());
		}
		else {
			Double value = map.get(b);
			if (value != null) return value.doubleValue();
		}
		
		double value = this.calc(a, b);
		map.put(b, new Double(value));
		return value;
	}
	
	/**
	 * Applies the kernel to two vectors.
	 * @param a Vector one.
	 * @param b Vector two.
	 * @return Output of the kernel function.
	 */
	public abstract double calc(RealVector a, RealVector b) throws KernelException;
	
}
