package org.singularity.algorithms;

/**
 * Common exception class for all learning errors.     
 */
public class LearningException extends Exception {

	/** Unique ID for serialization. */
	private static final long serialVersionUID = -7665018876720079374L;

	/**
	 * Creates a new learning exception.
	 * @param message Error message.
	 */
	public LearningException(String message) {
		super(message);
	}

	/**
	 * Creates a new learning exception.
	 * @param message Error message.
	 * @param cause Underlying error.
	 */
	public LearningException(String message, Throwable cause) {
		super(message, cause);
	}

}
