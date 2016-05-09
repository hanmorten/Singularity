package org.singularity.algorithms.kernel;

/**
 * Exception for all kernel operation errors.     
 */
public class KernelException extends Exception {

	/**
	 * Creates a new kernel exception.
	 * @param message Error message.
	 */
	public KernelException(String message) {
		super(message);
	}
	
	/**
	 * Creates a new kernel exception.
	 * @param message Error message.
	 * @param cause Underlying error.
	 */
	public KernelException(String message, Throwable cause) {
		super(message, cause);
	}

}
