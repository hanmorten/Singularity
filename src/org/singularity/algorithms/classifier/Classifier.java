package org.singularity.algorithms.classifier;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.LearningAlgorithm;
import org.singularity.algorithms.LearningListener;
import org.singularity.algorithms.TrainingSample;
import org.singularity.algorithms.TrainingSet;

/**
 * Base class for all binary classification learning algorithms.
 */
public abstract class Classifier implements LearningAlgorithm {

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
	 * Trains the classifier using a set of training samples.
	 * @param samples Training samples.
	 * @throws ClassifierException on any error.
	 */
	public abstract void train(TrainingSet samples) throws ClassifierException;

	public abstract double real(RealVector x) throws ClassifierException;

	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * @param input Feature vector of test sample.
	 * @return -1 or +1.
	 * @throws ClassifierException on any error.
	 */
	public abstract double test(RealVector input) throws ClassifierException;
	
	/**
	 * Computes the accuracy of the classifier (post training).
	 * @param samples Training samples with assigned labels.
	 * @return Accuracy on a scale from 0.0 to 1.0.
	 * @throws ClassifierException on any testing error.
	 */
	public double accuracy(TrainingSet samples) throws ClassifierException {
		if (samples.size() == 0) return 0.0d;
		double correct = 0;
		for (int i=0; i<samples.size(); i++) {
			final TrainingSample sample = samples.get(i);
			final double result = this.test(sample.getFeatures());
			if (result == sample.getLabel())
				correct++;
		}
		return correct / (double)samples.size();
	}

	/**
	 * Very basic plotting of 2-dimensional problem sets.
	 * @throws ClassifierException on testing errors.
	 */
	public void plot2D() throws ClassifierException {
		final double[] data = new double[2];
		System.out.println("    -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9 ");
		for (int x1 = 9; x1>-10; x1--) {
			System.out.print(" ");
			if (x1 >= 0) System.out.print(" ");
			System.out.print(x1);
			System.out.print(" ");
			for (int x2 = -9; x2<10; x2++) {
				data[0] = x1;
				data[1] = x2;
				if (this.test(new ArrayRealVector(data)) < 0)
					System.out.print("   ");
				else
					System.out.print(" # ");
			}
			System.out.println();
		}	
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
