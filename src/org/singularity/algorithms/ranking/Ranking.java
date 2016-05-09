package org.singularity.algorithms.ranking;

import java.util.List;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.LearningAlgorithm;
import org.singularity.algorithms.LearningException;
import org.singularity.algorithms.LearningListener;
import org.singularity.algorithms.TrainingSet;

/**
 * Common interface for all ranking implementations.  
 */
public abstract class Ranking implements LearningAlgorithm, LearningListener {

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
	 * Trains the algorithm using a set of training samples.
	 * Included for completion - do not use for ranking.
	 * @param samples Training samples.
	 * @throws RankingException on any training error.
	 */
	public void train(TrainingSet samples) throws RankingException {
		throw new RankingException("Do not use this method for ranking. Use addSample() method and then train().");
	}

	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * Included for completion - do not use for ranking.
	 * @param input Feature vector of test sample.
	 * @return predicted output for input feature vector.
	 * @throws LearningException on any testing error.
	 */
	public double test(RealVector input) throws RankingException {
		throw new RankingException("Do not use this method for ranking. Use sort() instead.");
	}

	/**
	 * Computes the accuracy of the classifier (post training).
	 * @return Accuracy on a scale from 0.0 to 1.0.
	 * @throws ClassifierException on any testing error.
	 */
	public abstract double accuracy() throws RankingException;

	/**
	 * Computes the accuracy of the classifier (post training).
	 * @param samples Training samples to test against (ignored).
	 * @return Accuracy on a scale from 0.0 to 1.0.
	 * @throws ClassifierException on any testing error.
	 */
	public double accuracy(TrainingSet samples) throws RankingException {
		return this.accuracy();
	}
	
	
	/**
	 * Records a subjects preference of one object over another.
	 * @param subject Subject whose preference we're recording.
	 * @param a Undesired object.
	 * @param b Preferred object.
	 */
	public abstract void addSample(RankingSubject subject, RankingObject a, RankingObject b);

	/**
	 * Trains the underlying learning algorithm using the recorded subject
	 * preferences passed to the addSample() method.
	 * @throws RankingException on any error.
	 */
	public abstract void train() throws RankingException;

	/**
	 * Finds the best order between two ranking objects. 
	 * @param subject Subject that the order is according to.
	 * @param a First ranking object.
	 * @param b Second ranking object.
	 * @return -1.0 if object a should be before object b, 1.0 otherwise.
	 * @throws RankingException on any ranking error.
	 */
	public abstract double test(RankingSubject subject, RankingObject a, RankingObject b) throws RankingException;

	/**
	 * Sorts a list of ranking objects based on a given subject's predicted
	 * preference.
	 * @param subject Subject, observer for ranking (this can be a person with
	 *    a specific set of predicted preferences, etc).
	 * @param objects A list of object to sort based on subject's preferences.
	 */
	public abstract void sort(RankingSubject subject, List<RankingObject> objects) throws RankingException;
	
	/**
	 * Indicates that training is starting.
	 * @param algorithm Learning algorithm instance.
	 */
	public void trainingStart(LearningAlgorithm algorithm) {
		if (this.listener != null) this.listener.trainingStart(this);
	}
	
	/**
	 * Callback that is invoked for each training iteration. Note that some
	 * learning algorithms do not have a fixed number of iterations, and for
	 * these algorithms the 'total' parameter will be zero.
	 * @param algorithm Learning algorithm instance.
	 * @param iteration Iteration number.
	 * @param total Total number of iterations projected (or 0 if total is
	 *    unknown).
	 */
	public void trainingIteration(LearningAlgorithm algorithm, int iteration, int total) {
		if (this.listener != null) this.listener.trainingIteration(this, iteration, total);
	}
	
	/**
	 * Indicates that training is complete.
	 * @param algorithm Learning algorithm instance.
	 */
	public void trainingEnd(LearningAlgorithm algorithm, TrainingSet samples) {
		if (this.listener != null) this.listener.trainingEnd(this, samples);
	}

}
