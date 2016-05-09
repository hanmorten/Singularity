package org.singularity.algorithms.regression;

import org.singularity.algorithms.LearningException;

/**
 * Exception used for all training and testing errors.     
 */
public class RegressionException extends LearningException {

	/** Unique id for serialization. */
	private static final long serialVersionUID = -2856480254632648751L;

	/**
	 * Creates a new regression exception.
	 * @param message Error message.
	 */
	public RegressionException(String message) {
		super(message);
	}

	/**
	 * Creates a new regression exception.
	 * @param message Error message.
	 * @param cause Underlying error.
	 */
	public RegressionException(String message, Throwable cause) {
		super(message, cause);
	}

}
