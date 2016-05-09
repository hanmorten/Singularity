package org.singularity.application.forecast;

import java.text.NumberFormat;

import org.singularity.algorithms.LearningException;
import org.singularity.algorithms.clustering.ExpectationMaximization;

/**
 * Forecasts the unconstrained demand for a product/service for a specific
 * point in time.
 */
public class UnconstrainedDemand {

	/** Daily product demand (combination of recorded and projected values). */
	private int[] demand = null;
	/** Accumulated product demand (combination of recorded and projected values). */
	private int[] accumulated = null;
	/** Number of daily/accumulated demand values that are actual/recorded. */
	private int recorded = 0;
	/** Number of daily/accumulated demand values that are projections. */
	private int projected = 0;
	/** Product constraint (inventory available on consumption day). */
	private int constraint = 0;
	
	/**
	 * Forecasts unconstrained demand curve for a product, given a set of
	 * daily sales/demand figures.
	 * @param start Start day of provided, represented by a negative integer
	 *    which is -N days before the consumption date of the product.  
	 * @param demand Sales/demand figures for the product from day -N up to
	 *    any point before or on the consumption date.
	 * @param constraint Number of products available at consumption date.
	 */
	public UnconstrainedDemand(int start, int[] demand, int constraint) throws LearningException {
		this.constraint = constraint;
	
		// Result array that we'll return. This contains aggregated booking
		// data, up to and including the consumption date (day 0).
		this.demand = new int[1 - start];
		this.accumulated = new int[1 - start];

		// Find point when constrained demand was met. Sales figures after
		// this date will always be zero, so we'll discard these and use the
		// trained model to estimate what these figures would have been like.
		int max = demand.length;
		int accumulated = 0;
		for (int i=0; i<demand.length; i++) {
			accumulated += demand[i];
			if (accumulated >= constraint) {
				max = i;
				break;
			}
		}
		
		// Store the unconstrained booking data in a data-model that the EM
		// algorithm will understand.
		final double[][] bookings = new double[max][2];
		for (int i=0; i<max; i++) {
			bookings[i][0] = i + start;
			bookings[i][1] = demand[i];
		}
		
		// Train the EM algorithm.
		final ExpectationMaximization em = new ExpectationMaximization();
		em.train(bookings, 2);
		
		// Copy in the actual booking date for the dates we have figures for
		// and when demand was not constrained by product availability.
		for (int i=0; i<max; i++) {
			this.demand[i] = demand[i];
		}
		
		// Estimate the number of bookings we would get on the following days
		// if demand was not constrained by product availability.
		for (int i=max; i<(1-start); i++) {
			this.demand[i] = this.predict(em, i+start, constraint);
		}
		
		// Convert the array of daily bookings to an accumulated array.
		accumulated = 0;
		for (int i=0; i<this.demand.length; i++) {
			this.accumulated[i] = this.demand[i] + accumulated;
			accumulated = this.accumulated[i];
		}
		
		this.recorded = max;
		this.projected = this.demand.length - max; 
	}
	
	/**
	 * Returns the projected accumulated sales figures, which are a combination
	 * of the figures passed to the train() method and projected figures.
	 * @return accumulated demand curve for the product up to and including
	 *    the date of consumption.
	 */
	public int[] getAccumulatedForecast() {
		return this.accumulated;
	}
	
	/**
	 * Returns the projected daily sales figures, which are a combination of
	 * the figures passed to the train() method and projected figures.
	 * @return accumulated demand curve for the product up to and including
	 *    the date of consumption.
	 */
	public int[] getForecast() {
		return this.demand;
	}

	/**
	 * Returns the total number of products we think we can sell.
	 * @return the total number of products we think we can sell.
	 */
	public int getTotalSales() {
		return this.accumulated[this.accumulated.length - 1];
	}
	
	/**
	 * Returns the number of entries in the forecast that are real/recorded
	 * sales figures (as passed to the train() method).
	 * @return number of real/recorded sales figures in the forecast.
	 */
	public int getRecordedFigures() {
		return this.recorded;
	}
	
	/**
	 * Returns the number of entries in the forecast that are projected values.
	 * @return number of projected booking numbers in the forecast.
	 */
	public int getProjectedFigures() {
		return this.projected;
	}

	
	/**
	 * Returns the predicted booking number for a given day.
	 * @param day Day index (negative integer) before the consumption date.
	 * @param constraint Inventory cap/constraint.
	 * @return predicted number of bookings on this day.
	 */
	private int predict(ExpectationMaximization em, int day, int constraint) {
		double[] vector = new double[2];
		vector[0] = (double)day;
		double maxProbability = 0;
		int maxBookings = 0;
		
		// Find the daily booking number with the highest probability.
		for (int i=0; i<constraint * 2; i++) {
			vector[1] = (double)i;
			double probability = em.getDensity(vector);
			if (probability > maxProbability) {
				maxProbability = probability;
				maxBookings = i;
			}
		}
		
		return maxBookings;
	}
	
	/**
	 * Plots projected accumulated booking figures
	 */
	public void plot() {
		final NumberFormat nf = NumberFormat.getInstance();
		nf.setMaximumFractionDigits(0);
		nf.setMinimumFractionDigits(0);
		nf.setMaximumIntegerDigits(4);
		nf.setMinimumIntegerDigits(4);
		nf.setGroupingUsed(false);

		int max = 0;
		for (int i=0; i<this.accumulated.length; i++) {
			if (this.accumulated[i] > max) max = this.accumulated[i];
		}
		
		for (int i=max; i>0; i--) {
			System.out.print(nf.format(i));
			if (i == constraint) {
				System.out.print(" +-");
			}
			else {
				System.out.print(" | ");
			}
			for (int j=0; j<this.accumulated.length; j++) {
				if (i == constraint) {
					if (j == recorded)
						System.out.print("+");
					if (this.accumulated[j] >= i) {
						System.out.print("--#--");
					}
					else { 
						System.out.print("-----");
					}
				}
				else {
					if (j == recorded)
						System.out.print("|");
					if (this.accumulated[j] >= i) {
						System.out.print("  #  ");
					}
					else { 
						System.out.print("     ");
					}
				}
			}
			if (i == constraint) {
				System.out.println("-+");
			}
			else {
				System.out.println(" |");
			}
		}
		
		System.out.print("-----+-");
		for (int j=0; j<this.accumulated.length; j++) {
			if (j == recorded)
				System.out.print("+");
			System.out.print("-----");
		}
		System.out.println("-+");

		System.out.print("Day  | ");
		for (int j=0; j<this.accumulated.length; j++) {
			if (j == recorded)
				System.out.print("|");
			String value = " " + new Integer(1 + j - this.accumulated.length).toString();
			while (value.length() < 5) value = value + " ";
			System.out.print(value);
		}
		System.out.println(" |");

		System.out.print("     | ");
		String left = "Recorded ";
		while (left.length() < 5 * this.recorded) left = " "+left+" ";
		String right = "Projected ";
		while (right.length() < 5 * this.projected) right = " "+right+" ";
		System.out.println(left+"|"+right+" |");
	}
	
	/**
	 * Test method.
	 * @param args Command-line arguments (none).
	 */
	public static void main(String[] args) {
		try {
			// Our recorded daily booking figures, starting on day -20
			// (20 days before consumption).
			final int[] demand = { 1, 1, 0, 1, 0, 0, 1, 1, 1, 2, 1 };

			// Estimate the unconstrained demand.
			final UnconstrainedDemand ud = new UnconstrainedDemand(-20, demand, 20);
			// This array starts on day -20 and ends on day 0.
			final int[] unconstrained = ud.getAccumulatedForecast();
			
			// Plot the unconstrained demand in the console (very basic)...
			ud.plot();
			
			// Find the maximum number of products we think we can sell.
			final int products = ud.getTotalSales();
			System.out.println("\nWith unlimited inventory, we can sell "+products+" products.\n\n");
		}
		catch (Throwable e) {
			System.err.println(e.getMessage());
			e.printStackTrace(System.err);
		}
	}

}
