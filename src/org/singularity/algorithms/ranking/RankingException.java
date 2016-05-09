package org.singularity.algorithms.ranking;

import org.singularity.algorithms.LearningException;

/**
 * Exception used for all ranking training and testing errors.     
 */
public class RankingException extends LearningException {

	/** UID required for serialization. */
	private static final long serialVersionUID = 6992068909046145977L;

	/**
	 * Creates a new classifier exception.
	 * @param message Error message.
	 */
	public RankingException(String message) {
		super(message);
	}

	/**
	 * Creates a new classifier exception.
	 * @param message Error message.
	 * @param cause Underlying error.
	 */
	public RankingException(String message, Throwable cause) {
		super(message, cause);
	}

}
