package org.singularity.application.calendar;

import java.util.*;

import org.apache.commons.math3.linear.RealVector;

import org.singularity.algorithms.TrainingSet;
import org.singularity.algorithms.TrainingSample;
import org.singularity.algorithms.regression.RegressionException;
import org.singularity.algorithms.regression.PoissonRegression;

/**
 * Predicts magnitudes (such as sales figures) for specific calendar days.
 */
public class CalendarPredictor {

	/** This object converts Calendar events to feature vectors. */
	private CalendarEvents events = new CalendarEvents();
	
	/** Poisson regression algorithm we'll use to make predictions. */
	private PoissonRegression regression = new PoissonRegression(100);
	
	/**
	 * Creates a new calendar predictor.
	 */
	public CalendarPredictor() {
		
	}
	
	/**
	 * Trains the predictor using events recorded in the past.
	 * @param from Start date for recored events.
	 * @param count Array of daily counts/mangnitudes/sales, starting on the
	 *    given start date.
	 * @throws RegressionException if the Poisson algorithm cannot be trained.
	 */
	public void train(Calendar from, double[] count) throws RegressionException {
		final TrainingSet samples = new TrainingSet();
		
		final Calendar date = (Calendar)from.clone();
		
		for (int i=0; i<count.length; i++) {
			final RealVector features = events.getFeatureVector(date);
			final double label = count[i];
			final TrainingSample sample = new TrainingSample(features, label);
			samples.add(sample);
		}
		
		this.regression.train(samples);
	}
	
	/**
	 * Predicts a daily count/mangnitude/sales for a given date.
	 * @param date Date to make prediction for.
	 * @return predicted number of counts/sales/etc.
	 * @throws RegressionException if the Poisson algorithm could not make
	 *   a prediction (for any reason).
	 */
	public double predict(Calendar date) throws RegressionException {
		final RealVector features = events.getFeatureVector(date);
		return this.regression.test(features);
	}

}
