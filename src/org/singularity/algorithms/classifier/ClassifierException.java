package org.singularity.algorithms.classifier;

import org.singularity.algorithms.LearningException;

/**
 * Exception used for all training and testing errors.     
 */
public class ClassifierException extends LearningException {

	/** UID required for serialization. */
	private static final long serialVersionUID = 6992068909046145977L;

	/**
	 * Creates a new classifier exception.
	 * @param message Error message.
	 */
	public ClassifierException(String message) {
		super(message);
	}

	/**
	 * Creates a new classifier exception.
	 * @param message Error message.
	 * @param cause Underlying error.
	 */
	public ClassifierException(String message, Throwable cause) {
		super(message, cause);
	}

}
