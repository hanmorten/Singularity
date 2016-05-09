package org.singularity.algorithms.regression;

import org.apache.commons.math3.linear.*;
import org.singularity.*;
import org.singularity.algorithms.LearningAlgorithm;
import org.singularity.algorithms.LearningListener;
import org.singularity.algorithms.TrainingSample;
import org.singularity.algorithms.TrainingSet;

/**
 * Base class for all regression implementations.     
 */
public abstract class Regression implements LearningAlgorithm {

	/** Learning listener for progress callbacks. */
	protected LearningListener listener = null;
	
	/**
	 * Passes an implementation of the LearningListener interface to the
	 * learning algorithm, so that the learning algorithm can feed learning
	 * process information to the listener.
	 * @param listener Learning listener callback implementation.
	 */
	public void setLearningListener(LearningListener listener) {
		this.listener = listener;
	}
	
	/**
	 * Trains the regression algorithm using a set of training samples.
	 * @param samples Training samples.
	 * @throws RegressionException on any error.
	 */
	public abstract void train(TrainingSet samples) throws RegressionException;

	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * @param input Feature vector of test sample.
	 * @return real value result
	 * @throws RegressionException on any error.
	 */
	public abstract double test(RealVector input) throws RegressionException;
	
	/**
	 * Computes the accuracy of the regression algorithm (post training).
	 * @param samples Training samples with assigned labels.
	 * @return Accuracy on a scale from 0.0 to 1.0.
	 * @throws RegressionException on any testing error.
	 */
	public double accuracy(TrainingSet samples) throws RegressionException {
		if (samples.size() == 0) return 0.0d;
		double diff = 0;
		for (int i=0; i<samples.size(); i++) {
			final TrainingSample sample = samples.get(i);
			final double result = this.test(sample.getFeatures());
			diff += Math.abs(result - sample.getLabel()); 
		}
		return 1 - (diff / (double)samples.size());
	}

	/**
	 * Returns the name of the learning algorithm implemented by this class.
	 * @return the name of the learning algorithm implemented by this class.
	 */
	public String getName() {
		final String name = this.getClass().getName();
		final int dot = name.lastIndexOf('.');
		if (dot > -1)
			return name.substring(dot+1);
		else
			return name;
		
	}

}
