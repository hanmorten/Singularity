package org.singularity.algorithms.regression;

import java.util.LinkedList;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.LearningListenerStdout;
import org.singularity.algorithms.TrainingSample;
import org.singularity.algorithms.TrainingSet;

public class PoissonRegression extends Regression implements java.io.Serializable {

	/** Unique ID for serialization. */
	private static final long serialVersionUID = -1031977336131750123L;

	/**
	 * Implementation of a linked list with finite size.
	 */
	private class FiniteLinkedList<T> extends LinkedList<T> {

		/** Unique ID for serialization. */
		private static final long serialVersionUID = 4530054182676011223L;

		/** Maximum size of the list. */
		private int maxSize = 4;
		
		/**
		 * Creates a new linked list with finite size.
		 * @param maxSize Max allowed size of list.
		 */
		public FiniteLinkedList(int maxSize) {
			this.maxSize = maxSize;
		}

		public boolean add(T data) {
			if (this.size() == maxSize) {
				super.removeFirst();
				return super.add(data);
			}
			else {
				return super.add(data);
			}
		}
		
	}

	/**
	 * Optimizes the line for the Poisson regression algorithm.
	 */
	private class StepSearcher  {

		/** Reference to owner Poisson regression algorithm. */
		private PoissonRegression regression;

		/** Cut-off point for maximum step size. */
		private double maxStep = 100;

		/** Termination condition: Iteration stops after number of iterations
		 * has reached this value.
		 */
		private int maxIterations = 100;

		/**
		 * Termination condition: Iteration stops if abs(delta(x/x)) is less
		 * than the value of this parameter.
		 */
		private double relativeTolerance = 1e-7;
		
		/**
		 * Termination condition: Iteration stops if abs(delta(x)) is less
		 * than the value of this parameter.
		 */
		private double absoluteTolerance = 1e-4;

		/**
		 * Termination condition: Iteration stops if the function 
		 * increased sufficiently (Wolf condition).
		 */
		private double ALF = 1e-4;

		public StepSearcher(PoissonRegression regression) {
			this.regression = regression;
		}

		/**
		 * Determines the next step to take in the given direction, and
		 * applies to the beta parameters of the regression algorithm.
		 * If no good step can be found, the beta parameters are left as
		 * they are.
		 * @param direction Current direction vector.
		 * @return true if a good step was found.
		 * @throws RegressionException on any error.
		 */
		public boolean optimize(RealVector direction) throws RegressionException  {
			try {
				RealVector oldParameters = (RealVector)regression.getParameters().copy();
				RealVector parameters = (RealVector)regression.getParameters();
				RealVector gradient = (RealVector)regression.getGradient().copy();

				final double sum = direction.getNorm();
				if (sum > maxStep) {
					direction.mapMultiplyToSelf(maxStep/sum);
				}

				final double slope = gradient.dotProduct(direction);
				if (slope <= 0) return false;

				// Find maximum lambda when:
				//   delta(x) / x < relativeTolerance for all coordinates.
				// The largest step size that triggers this threshold is
				// stored in the in the origLambda.
				double test = 0.0;
				for (int i=0; i<oldParameters.getDimension(); i++) {
					double temp = Math.abs(direction.getEntry(i)) / Math.max(Math.abs(oldParameters.getEntry(i)), 1.0);
					if (temp > test) test = temp;
				}
				final double origLambda = relativeTolerance / test;

				double oldFunc = regression.getValue();
				final double origFunc = oldFunc;
				
				double lambda  = 1.0;
				double oldLambda = 0.0;
				double lambda2 = 0.0;
				double tempLambda = 0.0; 

				// look for step size in direction given by "line"
				for (int iteration=0; iteration < maxIterations; iteration++) {
					parameters.combineToSelf(1, lambda - oldLambda, direction);

					// check for convergence 
					//convergence on delta x
					if ((lambda < origLambda) || checkConvergence(oldParameters, parameters)) {
						regression.setParameters(oldParameters);
						return false;
					}

					oldLambda = lambda;
					
					final double func = regression.getValue();

					// Check for sufficient function increase (Wolf condition)
					if (func >= origFunc + ALF * lambda * slope) { 
						if (func < origFunc) 
							throw new RegressionException("Function did not increase: f=" + func + " < " + origFunc + "=fold");				
						return true;
					}
					// if value is infinite, i.e. we've
					// jumped to unstable territory, then scale down jump
					else if (Double.isInfinite(func) || Double.isInfinite(oldFunc)) {
						tempLambda = 0.2 * lambda;					
						if (lambda < origLambda) { //convergence on delta x
							regression.setParameters(oldParameters);
							return false;
						}
					}
					else { // backtrack
						if (lambda == 1.0) {// first time through
							tempLambda = -slope/(2.0*(func-origFunc-slope));
						}
						else {
							double rhs1 = func - origFunc - lambda * slope;
							double rhs2 = oldFunc - origFunc - lambda2 * slope;
							double a = (rhs1/(lambda*lambda)-rhs2/(lambda2*lambda2)) / (lambda - lambda2);
							double b = (-lambda2*rhs1/(lambda*lambda)+lambda*rhs2/(lambda2*lambda2)) / (lambda - lambda2);
							if (a == 0.0) {
								tempLambda = -slope/(2.0*b);
							}
							else {
								final double disc = b * b - 3.0 * a * slope;
								if (disc < 0.0) {
									tempLambda = .5 * lambda;
								}
								else if (b <= 0.0) {
									tempLambda = (-b+Math.sqrt(disc))/(3.0*a);
								}
								else {
									tempLambda = -slope/(b+Math.sqrt(disc));
								}
							}

							if (tempLambda > .5 * lambda) {
								tempLambda = .5 * lambda;
							}
						}
					}
					
					lambda2 = lambda;
					oldFunc = func;
					lambda = Math.max(tempLambda, .1 * lambda);						
				}

				return false;
			}
			catch (Exception e) {
				throw new RegressionException("Error determining next step: "+e.getMessage(), e);
			}
		}

		/**
		 * Checks for convergence, and returns true if we've converged. The
		 * check is essentially to see if the difference of each entry in the
		 * parameters vs. the old parameters are all within the absolute
		 * tolerance.
		 * @param parameters New beta parameters for regression algorithm.
		 * @param oldParameters Old beta parameters for regression algorithm.
		 * @return true if we've converged, false otherwise.
		 */
		private boolean checkConvergence(RealVector parameters, RealVector oldParameters) {
			for (int i = 0; i < parameters.getDimension(); i++) {
				if (Math.abs(parameters.getEntry(i) - oldParameters.getEntry(i)) > absoluteTolerance) {
					return false;
				}
			}
			return true;
		}
	}

	/**
	 * Implementation of the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm
	 * for iteratively solving unconstrained nonlinear optimization problems.
	 * For details see:
	 * https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm
	 */
	private class NonlinearOptimizer {

		private RealVector gradient;
		private RealVector oldGradient;
		private RealVector direction;
		private RealVector parameters;
		private RealVector oldParameters;

		private double[] alpha;
		private int iterations;
		private int maxIterations = 1000;	
		private double tolerance = 0.0001;

		private double gradientTolerance = 0.001;
		private double epsilon = 1.0e-5;

		/**
		 *  The number of corrections used in BFGS update.
		 *  Value should be between 3 and 7.
		 */
		private int m = 7;

		/** BFGS: Finite list of m previous "parameters" values. */
		private LinkedList<RealVector> s = new FiniteLinkedList<RealVector>(m);
		/** BFGS: Finite list of m previous "gradient" values. */
		private LinkedList<RealVector> y = new FiniteLinkedList<RealVector>(m);
		/** BFGS: Finite list of SY values. */
		private LinkedList<Double> rho = new FiniteLinkedList<Double>(m);

		private PoissonRegression regression;

		final StepSearcher stepper;
		
		public NonlinearOptimizer(PoissonRegression regression) throws RegressionException {
			this.regression = regression;
			this.stepper = new StepSearcher(regression);
			this.initialize();
		}

		public boolean initialize() throws RegressionException {
			this.iterations = 0;
			this.alpha = new double[m];	    
			for (int i=0; i<m; i++)
				this.alpha[i] = 0.0;

			this.oldParameters = (RealVector)regression.getParameters().copy();
			this.gradient = regression.getGradient();
			this.oldGradient = (RealVector)gradient.copy();
			this.direction = (RealVector)gradient.copy();

			if (this.direction.getL1Norm() == 0) {
				this.gradient = null;
				return false;
			}

			// Calculate the first direction to take. 
			this.direction.mapMultiplyToSelf(1.0 / this.direction.getNorm());

			// Ask the step optimizer to find the next best step and
			// apply it to the regression algorithm. If the optimizer
			// could not find a good step then we bail at this point.
			if (!stepper.optimize(this.direction)) {
				this.gradient = null;
				return false;
			}

			this.parameters = (RealVector)regression.getParameters();
			this.gradient = regression.getGradient();
			return true;
		}

		public boolean optimize(int numIterations) throws RegressionException {

			try {

				if (this.gradient == null) {
					if (!this.initialize()) return false;
				}
				
				for (int iterationCount = 0; iterationCount < numIterations; iterationCount++)	{
					if (regression.listener != null) regression.listener.trainingIteration(regression, iterationCount, numIterations);

					final double value = regression.getValue();

					// get difference between previous 2 gradients and parameters
					for (int i=0; i < parameters.getDimension(); i++) {
						final double p = parameters.getEntry(i);
						final double op = oldParameters.getEntry(i);
						if (Double.isInfinite(p) &&	Double.isInfinite(op) && (p * op > 0))
							oldParameters.setEntry(i, 0.0d);
						else
							oldParameters.setEntry(i, p - op);

						final double g = gradient.getEntry(i);
						final double og = oldGradient.getEntry(i);
						if (Double.isInfinite(g) && Double.isInfinite(og) && (g * og > 0))
							oldGradient.setEntry(i, 0.0);
						else
							oldGradient.setEntry(i, gradient.getEntry(i) - oldGradient.getEntry(i));
					}
					
					for (int i=0; i<gradient.getDimension(); i++) {
						direction.setEntry(i,gradient.getEntry(i));
					}

					final double sy = oldParameters.dotProduct(oldGradient);
					if ( sy > 0 )
						throw new RegressionException ("sy = "+sy+" > 0" );

					final double gamma = sy / oldGradient.dotProduct(oldGradient);
					if (gamma > 0)
						throw new RegressionException("gamma = "+gamma+" > 0" );

					rho.add(1.0 / sy);
					s.add(oldParameters);
					y.add(oldGradient);

					// Calculate the new direction
					for (int i = s.size() - 1; i >= 0; i--) {
						alpha[i] =  rho.get(i).doubleValue() * s.get(i).dotProduct(direction);
						direction.combineToSelf(1, -1.0 * alpha[i], y.get(i));
					}

					direction.mapMultiplyToSelf(gamma);

					for (int i = 0; i < y.size(); i++) {
						double beta = rho.get(i).doubleValue() * y.get(i).dotProduct(direction);
						direction.combineToSelf(1, alpha[i] - beta, s.get(i));
					}

					oldParameters = parameters.copy();
					oldGradient = gradient.copy();
					direction.mapMultiplyToSelf(-1.0);
					
					// Ask the step optimizer to find the next best step and
					// apply it to the regression algorithm. If the optimizer
					// could not find a good step then we bail at this point.
					if (!stepper.optimize(direction)) { 
						return false;
					}

					// The line optimizer may have updated the parameters
					// (or betas) for the regression algorithm, so we need to
					// update our local copy.
					this.parameters = (RealVector)regression.getParameters();
					this.gradient = regression.getGradient();

					// Check if change in value is small enough to ignore.
					final double newValue = regression.getValue();
					if (2.0 * Math.abs(newValue-value) <= tolerance * (Math.abs(newValue) + Math.abs(value) + epsilon)) {
						return true;
					}

					// Check if gradient is small enough to ignore.
					final double gradientCheck = gradient.getNorm();
					if (gradientCheck < gradientTolerance || gradientCheck == 0.0d) {
						return true;
					}	    

					// Check if max iterations was reached.
					if (++iterations > maxIterations) {
						return true;
					}
				}
				return false;
			}
			catch (Exception e) {
				throw new RegressionException("Error optmizing non-linear function: "+e.getMessage(), e);
			}
		}

	}

	/** Number of training iterations to perform. */
	private int iterations = 100;
	/** Reference to samples to train the algorithm on. */ 
	private TrainingSet samples = null;
	/** The weights to use during testing of new samples. */
	private RealVector betas = new ArrayRealVector(1);
	/** Regularization (damping factor) for the algorithm. */
	private double regularization = 0.01;

	/**
	 * Creates a new Poisson regression learner.
	 * @param iterations Number of training iterations to perform.
	 */
	public PoissonRegression(int iterations) {
		this.iterations = iterations;
	}

	/**
	 * Trains the regression algorithm using a set of training samples.
	 * @param samples Training samples.
	 * @throws RegressionException on any error.
	 */
	public void train(TrainingSet samples) throws RegressionException {
		if (this.listener != null) this.listener.trainingStart(this);

		this.samples = samples;
		this.betas = new ArrayRealVector(samples.get(0).getFeatures().getDimension());

		new NonlinearOptimizer(this).optimize(this.iterations);

		this.samples = null;
		
		if (this.listener != null) this.listener.trainingEnd(this, samples);
	}

	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * @param v Feature vector of test sample.
	 * @return real value result
	 * @throws RegressionException on any error.
	 */
	public double test(RealVector v) throws RegressionException {
		try {
			return Math.exp(betas.dotProduct(v));
		}
		catch (Exception e) {
			throw new RegressionException("Unable to test using Poisson regression: "+e.getMessage(), e);
		}
	}

	private RealVector getParameters() {
		return this.betas;
	}

	private void setParameters(RealVector betas) {
		this.betas = betas;
	}

	private double getValue() throws Exception {
		double value = 0;
		for (int i = 0; i < samples.size(); i++) {
			final TrainingSample sample = samples.get(i);
			final RealVector x = sample.getFeatures();
			final double y = sample.getLabel();
			final double w = 1; // weight of training sample
			final double mu = betas.dotProduct(x);

			value += w * (-Math.exp(mu) + y * mu);
		}
		return value - regularization / 2 * betas.dotProduct(betas);
	}

	private RealVector getGradient() throws RegressionException {
		try {
			RealVector gradient = betas.mapMultiply(-regularization);
			for (int i=0; i <samples.size(); i++){
				final TrainingSample sample = samples.get(i);
				final RealVector x = sample.getFeatures();
				final double y = sample.getLabel();
				final double w = 1; // weight of training sample
				gradient = gradient.add(x.mapMultiply(w * (y - Math.exp(betas.dotProduct(x)))));
			}
			return gradient;
		}
		catch (Throwable e) {
			throw new RegressionException("Unable to determine gradient: "+e.getMessage(), e);
		}
	}

	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("PoissonRegression(Beta=[");
		for (int i=0; i<this.betas.getDimension(); i++) {
			if (i > 0) buf.append(", ");
			buf.append(this.betas.getEntry(i));
		}
		buf.append("])");
		return buf.toString();
	}

	/**
	 * Returns the name of the algorithm.
	 * @return the name of the algorithm.
	 */
	public String name() {
		return "PoissonRegression";
	}

	public static void main(String[] args) {
		try {
			final TrainingSet input = new TrainingSet();

			input.add(new double[] { 1,1 },  1.0);
			input.add(new double[] { 1,2 },  1.2);
			input.add(new double[] { 1,3 },  1.4);
			input.add(new double[] { 1,4 },  1.6);
			input.add(new double[] { 1,5 },  1.8);

			input.add(new double[] { 2,1 },  2.0);
			input.add(new double[] { 2,2 },  2.2);
			input.add(new double[] { 2,3 },  2.4);
			input.add(new double[] { 2,4 },  2.6);

			input.add(new double[] { 3,1 },  3.0);
			input.add(new double[] { 3,2 },  3.2);
			input.add(new double[] { 3,3 },  3.4);
			input.add(new double[] { 3,4 },  3.6);

			final Regression regression = new PoissonRegression(500);
			regression.setLearningListener(new LearningListenerStdout());
			regression.train(input);
			System.err.println("Accuracy: "+regression.accuracy(input));

			System.err.println("Predicted value for x=[1,1] is "+regression.test(new ArrayRealVector(new double[] { 1,1 })));
			System.err.println("Predicted value for x=[1,2] is "+regression.test(new ArrayRealVector(new double[] { 1,2 })));
			System.err.println("Predicted value for x=[1,3] is "+regression.test(new ArrayRealVector(new double[] { 1,3 })));
			System.err.println("Predicted value for x=[1,4] is "+regression.test(new ArrayRealVector(new double[] { 1,4 })));
			System.err.println("Predicted value for x=[1,5] is "+regression.test(new ArrayRealVector(new double[] { 1,5 })));
			System.err.println("Predicted value for x=[1,6] is "+regression.test(new ArrayRealVector(new double[] { 1,6 })));

			System.err.println("Predicted value for x=[2,1] is "+regression.test(new ArrayRealVector(new double[] { 2,1 })));
			System.err.println("Predicted value for x=[2,2] is "+regression.test(new ArrayRealVector(new double[] { 2,2 })));
			System.err.println("Predicted value for x=[2,3] is "+regression.test(new ArrayRealVector(new double[] { 2,3 })));
			System.err.println("Predicted value for x=[2,4] is "+regression.test(new ArrayRealVector(new double[] { 2,4 })));

			System.err.println("Predicted value for x=[3,1] is "+regression.test(new ArrayRealVector(new double[] { 3,1 })));
			System.err.println("Predicted value for x=[3,2] is "+regression.test(new ArrayRealVector(new double[] { 3,2 })));
			System.err.println("Predicted value for x=[3,3] is "+regression.test(new ArrayRealVector(new double[] { 3,3 })));
			System.err.println("Predicted value for x=[3,4] is "+regression.test(new ArrayRealVector(new double[] { 3,4 })));
			System.err.println("Predicted value for x=[3,5] is "+regression.test(new ArrayRealVector(new double[] { 3,5 })));
			System.err.println("Predicted value for x=[3,6] is "+regression.test(new ArrayRealVector(new double[] { 3,6 })));
			System.err.println("Predicted value for x=[3,7] is "+regression.test(new ArrayRealVector(new double[] { 3,7 })));
			System.err.println("Predicted value for x=[3,8] is "+regression.test(new ArrayRealVector(new double[] { 3,8 })));

			System.err.println("Predicted value for x=[4,1] is "+regression.test(new ArrayRealVector(new double[] { 4,1 })));
			System.err.println("Predicted value for x=[4,2] is "+regression.test(new ArrayRealVector(new double[] { 4,2 })));
			System.err.println("Predicted value for x=[4,3] is "+regression.test(new ArrayRealVector(new double[] { 4,3 })));
			System.err.println("Predicted value for x=[4,4] is "+regression.test(new ArrayRealVector(new double[] { 4,4 })));
		}
		catch (Throwable e) {
			System.err.println("Error: "+e.getMessage());
			e.printStackTrace(System.err);
		}
	}


}