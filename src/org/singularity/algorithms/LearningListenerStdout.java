package org.singularity.algorithms;

/**
 * TrainingListener implementation that dumps training process info to stdout.    
 */
public class LearningListenerStdout implements LearningListener {

	/** Timestamp for start of training. */
	private long start;
	
	/**
	 * Default constructor.
	 */
	public LearningListenerStdout() {
		
	}

	/**
	 * Indicates that training is starting.
	 * @param algorithm Learning algorithm instance.
	 */
	public void trainingStart(LearningAlgorithm algorithm) {
		//System.out.println("Starting training: "+algorithm);
		System.out.println("Starting training");
		
		this.start = System.currentTimeMillis();

		System.out.print("Training progress: ");
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
		System.out.print('#');
	}
	
	/**
	 * Indicates that training is complete.
	 * @param algorithm Learning algorithm instance.
	 */
	public void trainingEnd(LearningAlgorithm algorithm, TrainingSet samples) {
		//System.err.println("\nTraining complete: "+algorithm);
		System.err.println("\nTraining complete!");

		final long stop = System.currentTimeMillis();
		final long total = stop - this.start;
		long seconds = total / 1000;
		long millis = total % 1000;
		
		long minutes = seconds / 60;
		seconds = seconds % 60;
		
		long hours = minutes / 60;
		minutes = minutes % 60;
		
		final StringBuffer buf = new StringBuffer();
		buf.append(hours);
		buf.append(":");
		if (minutes < 10)
			buf.append("0");
		buf.append(minutes);
		buf.append(":");
		if (seconds < 10)
			buf.append("0");
		buf.append(seconds);
		buf.append(".");
		buf.append(millis);
		System.err.println("Trained in "+buf.toString());
	}

}
