package org.singularity.algorithms.ranking;

import java.util.*;

import org.apache.commons.math3.linear.*;
import org.singularity.*;
import org.singularity.algorithms.TrainingSet;
import org.singularity.algorithms.classifier.Boosting;
import org.singularity.algorithms.classifier.Classifier;
import org.singularity.algorithms.classifier.ClassifierException;

/**
 * Implementation of pairwise ranking using a supporting binary
 * classification learning algorithm.     
 */
public class PairwiseRanking extends Ranking implements java.io.Serializable {

	/** UID required for serialization. */
	private static final long serialVersionUID = 6693887080954247162L;

	/** Classifier used for ranking. */
	private Classifier classifier;

	/** Training set that we accumulate. */
	private transient TrainingSet set;
	
	/** Accuracy of the classifier. */
	private double accuracy;

	/**
	 * Creates a new ranking classifier.
	 * @param classifier Binary classifier to use for comparisons.
	 */
	public PairwiseRanking(Classifier classifier) {
		this.classifier = classifier;
		this.set = new TrainingSet();
	}

	/**
	 * Records a subjects preference of one object over another.
	 * @param subject Subject whose preference we're recording.
	 * @param a Undesired object.
	 * @param b Preferred object.
	 */
	public void addSample(RankingSubject subject, RankingObject a, RankingObject b) {
		RealVector merged = subject.getFeatures().append(a.getFeatures()).append(b.getFeatures());
		this.set.add(merged, +1.0);
		merged = subject.getFeatures().append(b.getFeatures()).append(a.getFeatures());
		this.set.add(merged, -1.0);
	}

	/**
	 * Trains the underlying classifier using the recorded subject preferences
	 * passed to the addSample() method.
	 * @throws RankingException
	 */
	public void train() throws RankingException {
		if (this.listener != null) this.classifier.setLearningListener(this);
		
		try {
			this.classifier.train(set);
			this.accuracy = this.classifier.accuracy(set);
		}
		catch (ClassifierException e) {
			throw new RankingException("Error training using Boosting algorithm: "+e.getMessage(), e);
		}
	}
	
	/**
	 * Finds the best order between two ranking objects. 
	 * @param subject Subject that the order is according to.
	 * @param a First ranking object.
	 * @param b Second ranking object.
	 * @return -1.0 if object a should be before object b, 1.0 otherwise.
	 * @throws RankingException on any ranking error.
	 */
	public double test(RankingSubject subject, RankingObject a, RankingObject b) throws RankingException {
		final RealVector merged = subject.getFeatures().append(a.getFeatures()).append(b.getFeatures());
		try {
			return this.classifier.test(merged);
		}
		catch (ClassifierException e) {
			throw new RankingException("Error testing using Boosting algorithm: "+e.getMessage(), e);
		}
	}
	
	/**
	 * Sorts a list of ranking objects based on a given subject's predicted
	 * preference.
	 * @param subject Subject, observer for ranking (this can be a person with
	 *    a specific set of predicted preferences, etc).
	 * @param objects A list of object to sort based on subject's preferences.
	 */
	public void sort(RankingSubject subject, List<RankingObject> objects) throws RankingException {
		new RankingSorter(subject, objects);
	}
	
	/**
	 * Computes the accuracy of the classifier (post training).
	 * @return Accuracy on a scale from 0.0 to 1.0.
	 * @throws ClassifierException on any testing error.
	 */
	public double accuracy() throws RankingException {
		return this.accuracy;
	}
	
	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("PairwiseRanking(");
		buf.append(this.classifier.toString());
		buf.append(")");
		return buf.toString();
	}

	/**
	 * Returns the name of the algorithm.
	 * @return the name of the algorithm.
	 */
	public String name() {
		return "PairwiseRanking("+this.classifier.name()+")";
	}

	/**
	 * Implementation of Comparator interface used to sort RankingObject
	 * instances using the trained classifier.
	 */
	private class RankingSorter implements Comparator<RankingObject> { 

		/** Feature vector template used during testing. */
		private double[] template;
		/** Offset into template feature vector for first object. */
		private int offset1;
		/** Offset into template feature vector for second object. */
		private int offset2;

		/**
		 * Sorts a list of ranking objects based on a given subject.
		 * @param subject Subject, observer for ranking (this can be a person with
		 *    a specific set of preferences, etc).
		 * @param objects A list of object to sort based on subject's preferences.
		 */
		public RankingSorter(RankingSubject subject, List<RankingObject> objects) {

			// Get the feature vector of the subject.
			final double[] sfeatures = subject.getFeatures().toArray();
			this.offset1 = sfeatures.length;

			// Get a sample of the feature vectors of the objects to sort.
			final double[] ofeatures = objects.get(0).getFeatures().toArray();
			this.offset2 = sfeatures.length + ofeatures.length;
			
			// Construct a template for the merged feature vector. 
			this.template = new double[sfeatures.length + ofeatures.length * 2];
			System.arraycopy(sfeatures, 0, template, 0, sfeatures.length);
			
			// Sort using this object as a comparator.
			Collections.sort(objects, this);
		}
		
		/*
		 * (non-Javadoc)
		 * @see java.util.Comparator#compare(java.lang.Object, java.lang.Object)
		 */
		public int compare(RankingObject a, RankingObject b) {
			// Get the feature vectors of the two objects.
			final double[] afeatures = a.getFeatures().toArray();
			final double[] bfeatures = b.getFeatures().toArray();

			// Merge the object features into the template
			System.arraycopy(afeatures, 0, this.template, offset1, afeatures.length);
			System.arraycopy(bfeatures, 0, this.template, offset2, bfeatures.length);
			
			// Use the ranking classifier to compare the two objects.
			try {
				if (classifier.test(new ArrayRealVector(this.template)) <= 0)
					return -1;
				else
					return 1;
			}
			catch (ClassifierException e) {
				return 0;
			}
		}

		/*
		 * (non-Javadoc)
		 * @see java.lang.Object#equals(java.lang.Object)
		 */
		public boolean equals(Object other) {
			return false;
		}

	}

}
