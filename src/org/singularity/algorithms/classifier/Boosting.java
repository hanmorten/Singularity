package org.singularity.algorithms.classifier;

import java.io.Serializable;
import java.util.List;
import java.util.ArrayList;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.LearningListenerStdout;
import org.singularity.algorithms.TrainingSample;
import org.singularity.algorithms.TrainingSet;

/**
 * Implements the Adpative Boost machine learning algorithm.     
 */
public class Boosting extends Classifier implements Serializable {

	/** UID required for serialization. */
	private static final long serialVersionUID = 8516830696563728236L;

	/** Default number of iterations to use. */
	private int iterations = 20;

	/**
	 * Weak base learner (decision stump) to use in ensemble. This weak
	 * learner works on a single dimension only. Within this dimension the
	 * learner also has an offset. The learner classifies all samples
	 * above the offset as -1or +1, where -1 or +1 is decided by the sign
	 * assigned to the learner. A series of learners are added to an
	 * ensemble, where each learner has an assigned weight/importance. 
	 */
	private class BaseLearner implements Serializable {

		/** UID required for serialisation. */
		private static final long serialVersionUID = -3470435654026999975L;

		/** Dimension the base learner works on. */
		private int dimension = 0;

		/** Offset within the dimension. */
		private double offset = 0.0d;

		/** Sign (output) of the base learner. */
		private double sign = 1.0d;
		/** Weight or importance assigned to this learner. */
		private double alpha = 1.0d;

		/**
		 * Creates a new base learner.
		 * @param sign Sign/output of this base learner.
		 * @param dimension The dimension this base learner works on.
		 * @param offset Offset or decision boundary within the dimension.
		 */
		public BaseLearner(double sign, int dimension, double offset) {
			this.sign = sign;
			this.dimension = dimension;
			this.offset = offset;
		}

		/**
		 * Runs a prediction based on a given vector of input features.
		 * @param input Test sample feature vector.
		 * @return -1 or +1.
		 */
		public double test(RealVector input) {
			double value = input.getEntry(dimension);
			if (value >= offset)
				return sign;
			else
				return 0.0 - sign;
		}

		/**
		 * Returns the weight/importance of this learner.
		 * @return the weight/importance of this learner.
		 */
		public double getAlpha() {
			return alpha;
		}

		/**
		 * Sets the weight/importance of this learner.
		 * @param alpha the weight/importance of this learner.
		 */
		public void setAlpha(double alpha) {
			this.alpha = alpha;
		}

		/*
		 * (non-Javadoc)
		 * @see java.lang.Object#toString()
		 */
		public String toString() {
			return "    Stump(Dimension="+this.dimension+",Sign="+this.sign+",Offset="+this.offset+",Alpha="+alpha+")";
		}

	}

	/**
	 * Ensemble of weak learners to use when making predictions. Within the
	 * ensemble, each weak learner has an assigned importance. The ensemble
	 * then makes predictions based on the sum of each weak learner's
	 * prediction multiplied with each learner's weight.
	 */
	private class Ensemble implements Serializable {

		/** UID required for serialisation. */
		private static final long serialVersionUID = 9006647954032999650L;

		/** List of weak learners within this ensemble. */
		private List<BaseLearner> ensemble = new ArrayList<BaseLearner>();

		/**
		 * Creates a new ensemble of weak learners.
		 */
		public Ensemble() {

		}

		/**
		 * Adds a new weak learner to the ensemble.
		 * @param stump New weak learner to add to ensemble.
		 */
		public void add(BaseLearner stump) {
			this.ensemble.add(stump);
		}

		/**
		 * Makes a prediction given an input feature vector.
		 * @param input test samble feature vector.
		 * @return -1 or +1.
		 */
		public double test(RealVector input) {
			if (this.real(input) > 0)
				return 1.0d;
			else
				return -1.0d;
		}

		/**
		 * Makes a prediction given an input feature vector.
		 * @param input test samble feature vector.
		 * @return test outcome
		 */
		public double real(RealVector input) {
			double result = 0.0d;
			for (int i=0; i<this.ensemble.size(); i++) {
				final BaseLearner stump = this.ensemble.get(i); 
				final double value = stump.test(input);
				final double alpha = stump.getAlpha();
				result += value * alpha;
			}
			return result;
		}

		/*
		 * (non-Javadoc)
		 * @see java.lang.Object#toString()
		 */
		public String toString() {
			final StringBuffer buf = new StringBuffer();
			buf.append("  Ensemble(\n");
			for (int i=0; i<this.ensemble.size(); i++) {
				final BaseLearner stump = this.ensemble.get(i);
				buf.append(stump.toString());
				if (i<this.ensemble.size() - 1)
					buf.append(',');
				buf.append('\n');
			}
			buf.append("  )");
			return buf.toString();
		}

	}

	/** Ensemble of weak learners. */
	private Ensemble ensemble = new Ensemble();

	/**
	 * Creates a new adaptive boosting learner.
	 * @param iterations Number of iterations to use.
	 */
	public Boosting(int iterations) {
		this.iterations = iterations;
	}

	/**
	 * Trains the adaptive boosting algorithm.
	 * @param samples Set of training samples.
	 */
	public void train(TrainingSet samples) throws ClassifierException {
		if (this.listener != null) this.listener.trainingStart(this);

		try {
			// Set the initial weight for all training samples
			final int count = samples.size();
			final double iweight = 1.0d / (double)count;
			for (int i=0; i<count; i++) {
				samples.get(i).setWeight(iweight);
			}

			// Perform the number of configured learning iterations.
			for (int i=0; i<iterations; i++) {
				if (this.listener != null) this.listener.trainingIteration(this, i, iterations);

				// Find the best decision stump.
				final BaseLearner stump = this.getBaseLearner(samples.getSamples());
				// Update the weights of all training samples.
				this.updateWeights(samples.getSamples(), stump);
				// Add the new stump to the ensemble.
				ensemble.add(stump);
			}
		}
		finally {
			if (this.listener != null) this.listener.trainingEnd(this, samples);
		}
	}

	/**
	 * Finds the best weak classifier for the training samples.
	 * @param samples Set of weighted training samples.
	 * @return the best weak classifier for the training samples.
	 */
	private BaseLearner getBaseLearner(List<TrainingSample> samples) {

		// Minimum error count so far
		double errors = Double.MAX_VALUE;
		// Sign for the split
		double sign = 1.0d;
		// Offset of sample where we put the split
		double  offset = 0;
		// Dimension the split
		int dim = 0;

		// Iterate over all dimensions
		final int fcount = samples.get(0).getFeatures().getDimension();
		for (int d = 0; d < fcount; d++) {

			// Iterate over all samples in the input set.
			for (int i = 0; i < samples.size(); i++) {

				// Get the current samples magnitude for this dimension
				final TrainingSample sample = samples.get(i);
				final double feature = sample.getFeatures().getEntry(d);

				// Error count/weights if new stump is positive.
				double errPos = 0.0d;
				// Error count/weights if new stump is negative.
				double errNeg = 0.0d;

				// Iterate over all other samples
				for (int j = 0; j < samples.size(); j++) {
					final TrainingSample other = samples.get(j);
					// If other sample has magnitude higher than this.
					if (other.getFeatures().getEntry(d) >= feature) {
						// Other label has value +1, so if the sign for a
						// stamp at this position was negative, this would
						// count as a classification error.
						if (other.getLabel() >= 0) {
							errNeg += other.getWeight();
						}
						// Other label has value -1, so if the sign for a
						// stamp at this position was positive, this would
						// count as an error.
						else {
							errPos += other.getWeight();
						}
					}
					// If other sample has magnitude lower than this.
					else {
						// Other label has value -1, so if the sign for a
						// stamp at this position was positive, this would
						// count as an error
						if (other.getLabel() < 0) {
							errNeg += other.getWeight();
						}
						else {
							errPos += other.getWeight();
						}
					}
				}

				// If a stump at this position, with a negative sign yields
				// the lowest error rate, then we set the stump here.
				if (errors > errNeg) {
					offset = feature;
					errors = errNeg;
					sign = -1.0d;
					dim = d;
				}

				// If a stump at this position, with a positive sign yields
				// the lowest error rate, then we set the stump here.
				if (errors > errPos) {
					offset = feature;
					errors = errPos;
					sign = 1.0d;
					dim = d;
				}
			}
		}

		// Create the new base learner stump
		final BaseLearner stump = new BaseLearner(sign, dim, offset);

		// Calculate the alpha (vote) for the new stump
		double alpha = 0.0d;
		if (errors > 0.0d)
			alpha = 0.5 * Math.log((1.0 - errors) / errors);
		else
			alpha = 1;
		stump.setAlpha(alpha);


		return stump;
	}

	/**
	 * Updates the weights of the training samples.
	 * @param samples Training samples we're updating.
	 * @param stump Decision stump that is about to be added to the ensemble.
	 */
	private void updateWeights(List<TrainingSample> samples, BaseLearner stump) {
		// Get the vote of the new decision stump.
		final double alpha = stump.getAlpha();

		// Used to accumulate the total weight across all samples.
		double sum = 0;

		// Get the weight adjustment multipliers for samples that are
		// correctly or incorrectly classified by the new weak learner.
		final double correct = Math.exp(-alpha);
		final double wrong = Math.exp(alpha);

		// Iterate over all training samples and adjust their weight.
		for (int i = 0; i < samples.size(); i++) {
			// Get the next sample and the value the new weak learner
			// predicts based on the sample's feature vector.
			final TrainingSample sample = samples.get(i);
			final double result = stump.test(sample.getFeatures());
			// Weak learner predicts correct result
			if (sample.getLabel() == result) {
				// This decreases the weight of the sample.
				sample.setWeight(sample.getWeight() * correct);
			}
			// Weak learner predicts incorrect result.
			else {
				// This increases the weight of the sample.
				sample.setWeight(sample.getWeight() * wrong);
			}
			// Add the sample's new weight to the accumulated total.
			sum += sample.getWeight();
		}

		// Make sure all the weights sum to 1.
		for (int i = 0; i < samples.size(); i++) {
			final TrainingSample sample = samples.get(i);
			sample.setWeight(sample.getWeight() / sum);
		}

	}

	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * @param input Feature vector of test sample.
	 * @return -1 or +1.
	 * @throws ClassifierException on any error.
	 */
	public double test(RealVector input) throws ClassifierException {
		return ensemble.test(input);
	}

	public double real(RealVector input) throws ClassifierException {
		return this.ensemble.real(input);
	}

	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("Boosting(\n");
		buf.append(ensemble);
		buf.append("\n)");
		return buf.toString();
	}

	/**
	 * Returns the name of the algorithm.
	 * @return the name of the algorithm.
	 */
	public String name() {
		return "Boosting";
	}


	public static void main(String[] args) {
		try {
			/*
			final TrainingSet input = new TrainingSet(new File("data.txt"), 500);
			System.err.println("LOADED!");
			 */
			final TrainingSet input = new TrainingSet();
			input.add(new double[] { -1,-1 },  1);
			input.add(new double[] { -1,-2 },  1);
			input.add(new double[] { -2,-1 },  1);
			input.add(new double[] { -2,-2 },  1);

			input.add(new double[] { 1,1 },  1);
			input.add(new double[] { 1,2 },  1);
			input.add(new double[] { 2,1 },  1);
			input.add(new double[] { 2,2 },  1);

			input.add(new double[] { 6,1 }, -1);
			input.add(new double[] { 7,1 }, -1);
			input.add(new double[] { 6,2 }, -1);
			input.add(new double[] { 7,2 }, -1);

			input.add(new double[] { 6,7 },  1);
			input.add(new double[] { 7,7 },  1);
			input.add(new double[] { 6,6 },  1);
			input.add(new double[] { 7,6 },  1);
			input.add(new double[] { 2,4 },  1);
			input.add(new double[] { 2,6 },  1);
			input.add(new double[] { 3,4 },  1);
			input.add(new double[] { 3,6 },  1);
			input.add(new double[] { 4,4 },  -1);
			input.add(new double[] { 4,6 },  -1);
			input.add(new double[] { 5,4 },  -1);
			input.add(new double[] { 5,6 },  -1);

			final Classifier classifier = new Boosting(25);
			classifier.setLearningListener(new LearningListenerStdout());
			classifier.train(input);
			System.err.println("Accuracy: "+classifier.accuracy(input));

			classifier.plot2D();

			System.err.println("Predicted value for x=[1,1] is "+classifier.test(new ArrayRealVector(new double[] { 1,1 })));
			System.err.println("Predicted value for x=[1,2] is "+classifier.test(new ArrayRealVector(new double[] { 1,2 })));
			System.err.println("Predicted value for x=[2,2] is "+classifier.test(new ArrayRealVector(new double[] { 2,2 })));
			System.err.println("Predicted value for x=[2,3] is "+classifier.test(new ArrayRealVector(new double[] { 2,3 })));
			System.err.println("Predicted value for x=[2,4] is "+classifier.test(new ArrayRealVector(new double[] { 2,4 })));
			System.err.println("Predicted value for x=[2,5] is "+classifier.test(new ArrayRealVector(new double[] { 2,5 })));
			System.err.println("Predicted value for x=[2,6] is "+classifier.test(new ArrayRealVector(new double[] { 2,6 })));
			System.err.println("Predicted value for x=[3,3] is "+classifier.test(new ArrayRealVector(new double[] { 3,3 })));
			System.err.println("Predicted value for x=[4,4] is "+classifier.test(new ArrayRealVector(new double[] { 4,4 })));
			System.err.println("Predicted value for x=[6,1] is "+classifier.test(new ArrayRealVector(new double[] { 6,1 })));
			System.err.println("Predicted value for x=[6,2] is "+classifier.test(new ArrayRealVector(new double[] { 6,2 })));
			System.err.println("Predicted value for x=[6,5] is "+classifier.test(new ArrayRealVector(new double[] { 6,5 })));
			System.err.println("Predicted value for x=[6,6] is "+classifier.test(new ArrayRealVector(new double[] { 6,6 })));
			System.err.println("Predicted value for x=[6,7] is "+classifier.test(new ArrayRealVector(new double[] { 6,7 })));
			System.err.println("Predicted value for x=[6,8] is "+classifier.test(new ArrayRealVector(new double[] { 6,8 })));

		}
		catch (Throwable e) {
			System.err.println("Error: "+e.getMessage());
			e.printStackTrace(System.err);
		}

	}
}
