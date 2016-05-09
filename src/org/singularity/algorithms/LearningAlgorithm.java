package org.singularity.algorithms;

import org.apache.commons.math3.linear.*;

/**
 * Common interface for all machine learning algorithms.     
 */
public interface LearningAlgorithm {

	/**
	 * Passes an implementation of the LearningListener interface to the
	 * learning algorithm, so that the learning algorithm can feed learning
	 * process information to the listener.
	 * @param listener Learning listener callback implementation.
	 */
	public void setLearningListener(LearningListener listener);
	
	/**
	 * Trains the algorithm using a set of training samples.
	 * @param samples Training samples.
	 * @throws LearningException on any training error.
	 */
	public void train(TrainingSet samples) throws LearningException;

	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * @param input Feature vector of test sample.
	 * @return predicted output for input feature vector.
	 * @throws LearningException on any testing error.
	 */
	public double test(RealVector input) throws LearningException;
	
	/**
	 * Computes the accuracy of the classifier (post training).
	 * @param samples Training samples with assigned labels.
	 * @return Accuracy on a scale from 0.0 to 1.0.
	 * @throws ClassifierException on any testing error.
	 */
	public double accuracy(TrainingSet samples) throws LearningException;
	
	/**
	 * Returns the name of the algorithm.
	 * @return the name of the algorithm.
	 */
	public String name();
	
}
