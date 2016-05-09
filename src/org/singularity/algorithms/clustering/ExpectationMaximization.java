package org.singularity.algorithms.clustering;

import java.util.*;

import org.singularity.algorithms.*;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.distribution.fitting.MultivariateNormalMixtureExpectationMaximization;
import org.apache.commons.math3.distribution.MixtureMultivariateNormalDistribution;

/**
 * Wrapper implementation of the Expectation-Maximzation (EM) algorithm,
 * piggybacking on the implementation from the Apache commons-math3 library.
 */
public class ExpectationMaximization {

	/** Result of training the model. */
	private MixtureMultivariateNormalDistribution model = null;
	
	/** The likelihood of the trained model being accurate. */
	private double logLikelihood = 0;
	
	/**
	 * Creates a new EM algorithm implementation.
	 */
	public ExpectationMaximization() {

	}

	public void train(double[][] data, int clusters) throws LearningException {
		try {
			final MultivariateNormalMixtureExpectationMaximization em =
				new MultivariateNormalMixtureExpectationMaximization(data);
			final MixtureMultivariateNormalDistribution model =
				MultivariateNormalMixtureExpectationMaximization.estimate(data, clusters);
			em.fit(model);
			this.model = em.getFittedModel();
			this.logLikelihood = em.getLogLikelihood();
		}
		catch (Throwable e) {
			throw new LearningException("Unable to build mixture model: "+e.getMessage(), e);
		}
	}

	public double getDensity(RealVector x) {
		return this.model.density(x.toArray());
	}
	
	public double getDensity(double[] x) {
		return this.model.density(x);
	}

	public double getLogLikeiehood() {
		return this.logLikelihood;
	}
	
	public static void main(String[] args) {
		try {
			final double[][] bookings = {
					{ -10, 0 },
					{ -9, 0 },
					{ -8, 1 },
					{ -7, 1 },
					{ -6, 1 },
					{ -5, 2 },
					{ -4, 2 },
					{ -3, 2 },
					{ -2, 3 }
			};
			
			final ExpectationMaximization em = new ExpectationMaximization();
			em.train(bookings, 2);
			System.err.println("CLUSTER -2=1: "+em.getDensity(new ArrayRealVector(new double[] { -2, 1 } )));
			System.err.println("CLUSTER -2=2: "+em.getDensity(new ArrayRealVector(new double[] { -2, 2 } )));
			System.err.println("CLUSTER -2=3: "+em.getDensity(new ArrayRealVector(new double[] { -2, 3 } )));
			System.err.println("CLUSTER -2=4: "+em.getDensity(new ArrayRealVector(new double[] { -2, 4 } )));
			System.err.println("CLUSTER -2=5: "+em.getDensity(new ArrayRealVector(new double[] { -2, 5 } )));
			System.err.println("CLUSTER -2=6: "+em.getDensity(new ArrayRealVector(new double[] { -2, 6 } )));
			System.err.println("CLUSTER -1=1: "+em.getDensity(new ArrayRealVector(new double[] { -1, 1 } )));
			System.err.println("CLUSTER -1=2: "+em.getDensity(new ArrayRealVector(new double[] { -1, 2 } )));
			System.err.println("CLUSTER -1=3: "+em.getDensity(new ArrayRealVector(new double[] { -1, 3 } )));
			System.err.println("CLUSTER -1=4: "+em.getDensity(new ArrayRealVector(new double[] { -1, 4 } )));
			System.err.println("CLUSTER -1=5: "+em.getDensity(new ArrayRealVector(new double[] { -1, 5 } )));
			System.err.println("CLUSTER -1=6: "+em.getDensity(new ArrayRealVector(new double[] { -1, 6 } )));
			System.err.println("CLUSTER  0=1: "+em.getDensity(new ArrayRealVector(new double[] { 0, 1 } )));
			System.err.println("CLUSTER  0=2: "+em.getDensity(new ArrayRealVector(new double[] { 0, 2 } )));
			System.err.println("CLUSTER  0=3: "+em.getDensity(new ArrayRealVector(new double[] { 0, 3 } )));
			System.err.println("CLUSTER  0=4: "+em.getDensity(new ArrayRealVector(new double[] { 0, 4 } )));
			System.err.println("CLUSTER  0=5: "+em.getDensity(new ArrayRealVector(new double[] { 0, 5 } )));
			System.err.println("CLUSTER  0=6: "+em.getDensity(new ArrayRealVector(new double[] { 0, 6 } )));
			System.err.println("CLUSTER  0=7: "+em.getDensity(new ArrayRealVector(new double[] { 0, 7 } )));

			System.err.println("CLUSTER  10=1: "+em.getDensity(new ArrayRealVector(new double[] { 10, 1 } )));
			System.err.println("CLUSTER  10=2: "+em.getDensity(new ArrayRealVector(new double[] { 10, 2 } )));
			System.err.println("CLUSTER  10=3: "+em.getDensity(new ArrayRealVector(new double[] { 10, 3 } )));
			System.err.println("CLUSTER  10=4: "+em.getDensity(new ArrayRealVector(new double[] { 10, 4 } )));
			System.err.println("CLUSTER  10=5: "+em.getDensity(new ArrayRealVector(new double[] { 10, 5 } )));
			System.err.println("CLUSTER  10=6: "+em.getDensity(new ArrayRealVector(new double[] { 10, 6 } )));
			System.err.println("CLUSTER  10=7: "+em.getDensity(new ArrayRealVector(new double[] { 10, 7 } )));
			System.err.println("CLUSTER  10=8: "+em.getDensity(new ArrayRealVector(new double[] { 10, 8 } )));
			System.err.println("CLUSTER  10=9: "+em.getDensity(new ArrayRealVector(new double[] { 10, 9 } )));
			System.err.println("CLUSTER  10=10: "+em.getDensity(new ArrayRealVector(new double[] { 10, 10 } )));
			System.err.println("CLUSTER  10=11: "+em.getDensity(new ArrayRealVector(new double[] { 10, 11 } )));
			System.err.println("CLUSTER  10=12: "+em.getDensity(new ArrayRealVector(new double[] { 10, 12 } )));
			System.err.println("CLUSTER  10=13: "+em.getDensity(new ArrayRealVector(new double[] { 10, 13 } )));
			System.err.println("CLUSTER  10=14: "+em.getDensity(new ArrayRealVector(new double[] { 10, 14 } )));
}
		catch (Throwable e) {
			e.printStackTrace();
		}

	}

}

