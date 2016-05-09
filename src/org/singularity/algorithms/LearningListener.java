package org.singularity.algorithms;

/**
 * Listener interface for all learning algorithms, providing feedback on
 * the progress of the training process.
 */
public interface LearningListener {

	/**
	 * Indicates that training is starting.
	 * @param algorithm Learning algorithm instance.
	 */
	public void trainingStart(LearningAlgorithm algorithm);
	
	/**
	 * Callback that is invoked for each training iteration. Note that some
	 * learning algorithms do not have a fixed number of iterations, and for
	 * these algorithms the 'total' parameter will be zero.
	 * @param algorithm Learning algorithm instance.
	 * @param iteration Iteration number.
	 * @param total Total number of iterations projected (or 0 if total is
	 *    unknown).
	 */
	public void trainingIteration(LearningAlgorithm algorithm, int iteration, int total);
	
	/**
	 * Indicates that training is complete.
	 * @param algorithm Learning algorithm instance.
	 * @param samples Training samples used during training.
	 */
	public void trainingEnd(LearningAlgorithm algorithm, TrainingSet samples);

}
