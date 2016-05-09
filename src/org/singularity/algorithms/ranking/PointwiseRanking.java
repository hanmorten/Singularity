package org.singularity.algorithms.ranking;

import java.util.*;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.TrainingSet;
import org.singularity.algorithms.regression.Regression;
import org.singularity.algorithms.regression.RegressionException;

/**
 * Implements pointwise ranking, using a regression learning algorithm to
 * generate the score for each sample.
 */
public class PointwiseRanking extends Ranking {

	private Regression regression;
	
	private TrainingSet samples;
	
	private double accuracy = 0.0;
	
	public PointwiseRanking(Regression regression) {
		this.regression = regression;
	}

	/**
	 * Computes the accuracy of the classifier (post training).
	 * @return Accuracy on a scale from 0.0 to 1.0.
	 * @throws ClassifierException on any testing error.
	 */
	public double accuracy() throws RankingException {
		return this.accuracy;
	}

	private class RankingObjectWrapper {
		
		private RealVector vector;
		private int hash = 0;
		
		public RankingObjectWrapper(RankingObject object) {
			this.vector = object.getFeatures();
			for (int i=0; i<this.vector.getDimension(); i++) {
				this.hash = this.hash ^ ((int)(this.vector.getEntry(i) * 1000.0) + i);
			}
		}
		
		public RealVector getVector() {
			return vector;
		}
		
		public int hashCode() {
			return this.hash;
		}

		public boolean equals(Object other) {
			return this.vector.equals(((RankingObjectWrapper)other).vector);
		}
		
	}

	private class RankingSubjectWrapper {
		
		private RealVector vector;
		private int hash = 0;
		
		/** Maps all objects that this subject has a view on to their score. */
		private Map<RankingObjectWrapper,Long> object2score =
				new HashMap<RankingObjectWrapper,Long>();

		/** Same set of objects, sorted by their score. */
		private TreeMap<Long,List<RankingObjectWrapper>> score2objects = 
				new TreeMap<Long,List<RankingObjectWrapper>>();
		
		
		public RankingSubjectWrapper(RankingSubject subject) {
			this.vector = subject.getFeatures();
			for (int i=0; i<this.vector.getDimension(); i++) {
				this.hash = this.hash ^ ((int)(this.vector.getEntry(i) * 1000.0) + i);
			}
		}
		
		private void add(RankingObjectWrapper object, long score) {
			final Long value = new Long(score);
			object2score.put(object, value);
			List<RankingObjectWrapper> list = score2objects.get(value);
			if (list == null) score2objects.put(value, list = new ArrayList<RankingObjectWrapper>());
			list.add(object);
		}
		
		private void inc(RankingObjectWrapper object) {
			final Long score = object2score.get(object);
			if (score == null) return;
			
			// Iterate the scores in descending order, meaning we'll
			// process the objects with the highest score first.
			Long oldScore = score2objects.lastKey();
			while (oldScore != null && oldScore.longValue() > score.longValue()) {
				// Get the list of objects with the current score.
				final List<RankingObjectWrapper> list = score2objects.get(oldScore);
				// Update each object with the new increased score.
				final Long newScore = new Long(oldScore.longValue() + 1);
				for (RankingObjectWrapper obj : list) {
					object2score.put(obj, newScore);
				}
				// Remove the objects from the old score.
				score2objects.remove(oldScore);
				// Add the objects to the new score.
				score2objects.put(newScore, list);
				
				oldScore = score2objects.lowerKey(oldScore);
			}
			
			// Remove main object from current score.
			List<RankingObjectWrapper> list = score2objects.get(score);
			for (int i=0; i<list.size(); i++) {
				final RankingObjectWrapper obj = list.get(i);
				if (obj.equals(object)) {
					list.remove(i);
					break;
				}
			}
			
			// Add main object with its new increased score.
			this.add(object, score.longValue() + 1);
		}
		
		private void dec(RankingObjectWrapper object) {
			final Long score = object2score.get(object);
			if (score == null) return;
			
			// Iterate the scores in ascending order, meaning we'll
			// process the objects with the highest score first.
			Long oldScore = score2objects.lastKey();
			while (oldScore != null && oldScore.longValue() < score.longValue()) {
				// Get the list of objects with the current score.
				final List<RankingObjectWrapper> list = score2objects.get(oldScore);
				// Update each object with the new lowered score.
				final Long newScore = new Long(oldScore.longValue() - 1);
				for (RankingObjectWrapper obj : list) {
					object2score.put(obj, newScore);
				}
				// Remove the objects from the old score.
				score2objects.remove(oldScore);
				// Add the objects to the new score.
				score2objects.put(newScore, list);
				
				oldScore = score2objects.higherKey(oldScore);
			}
			
			// Remove main object from current score.
			List<RankingObjectWrapper> list = score2objects.get(score);
			for (int i=0; i<list.size(); i++) {
				final RankingObjectWrapper obj = list.get(i);
				if (obj.equals(object)) {
					list.remove(i);
					break;
				}
			}
			
			// Add main object with its new increased score.
			this.add(object, score.longValue() - 1);
		}

		public void add(RankingObjectWrapper undesired, RankingObjectWrapper preferred) {
			
			final Long aScore = object2score.get(undesired);
			final Long bScore = object2score.get(preferred);
			
			// Neither the preferred, nor the undesired objects have a score,
			// so store them with scores of +1 and -1 respectively.
			if (aScore == null && bScore == null) {
				this.add(undesired, 0);
				this.add(preferred, +1);
			}
			else if (aScore == null) {
				// New entry for the undesired object.
				// Rank it lower than the preferred object's existing score
				this.add(undesired, bScore.longValue() - 1);
			}
			else if (bScore == null) {
				// New entry for the preferred object.
				// Rank it higher than the undesired object's existing score
				this.add(preferred, aScore.longValue() + 1);
			}
			// Both the undesired and the preferred objects exist, so increment
			// the score of the preferred object and decrement the score of the
			// undesired object.
			else {
				//this.dec(undesired);
				this.inc(preferred);
			}
			
		}
		
		public int hashCode() {
			return this.hash;
		}
		
		public boolean equals(Object other) {
			return this.vector.equals(((RankingSubjectWrapper)other).vector);
		}

		public void addTrainingSamples(TrainingSet set) {
			for (Long score : this.score2objects.keySet()) {
				final double label = (double)score;
				for (RankingObjectWrapper object : this.score2objects.get(score)) {
					final RealVector vector = this.vector.append(object.getVector());
					set.add(vector, label);
				}
			}
		}
		
		public String toString() {
			final StringBuffer buf = new StringBuffer();
			buf.append("RankingSubject");
			buf.append(this.vector);
			buf.append("(\n");
			for (Long score : this.score2objects.keySet()) {
				final double label = (double)score;
				for (RankingObjectWrapper object : this.score2objects.get(score)) {
					final RealVector vector = this.vector.append(object.getVector());
					buf.append("  ");
					buf.append(vector);
					buf.append(" = ");
					buf.append(label);
					buf.append("\n");
				}
			}
			buf.append(")");
			return buf.toString();
		}
	}

	/** Identity map for all ranking subjects. */
	private Map<RankingSubjectWrapper,RankingSubjectWrapper> subjects = 
			new HashMap<RankingSubjectWrapper,RankingSubjectWrapper>();
	
	private void dump() {
		for (RankingSubjectWrapper subject : this.subjects.keySet()) {
			System.err.println(subject);
		}
	}
	
	/**
	 * Records a subjects preference of one object over another.
	 * @param subject Subject whose preference we're recording.
	 * @param undesired Undesired object.
	 * @param preferred Preferred object.
	 */
	public void addSample(RankingSubject subject, RankingObject undesired, RankingObject preferred) {
		RankingSubjectWrapper s = new RankingSubjectWrapper(subject);
		if (subjects.containsKey(s))
			s = subjects.get(s);
		else
			subjects.put(s, s);
		final RankingObjectWrapper a = new RankingObjectWrapper(undesired);
		final RankingObjectWrapper b = new RankingObjectWrapper(preferred);
		s.add(a, b);
	}

	/**
	 * Trains the underlying regression algorithm using the recorded subject
	 * preferences passed to the addSample() method.
	 * @throws RankingException on any training error.
	 */
	public void train() throws RankingException {
		if (this.listener != null) this.regression.setLearningListener(this);

		try {
			// Create a new training set.
			this.samples = new TrainingSet();

			// Get all ranking subjects to add the feature vectors that
			// represents the preferred order of objects.
			for (RankingSubjectWrapper subject : this.subjects.keySet()) {
				subject.addTrainingSamples(this.samples);
			}
			
			// Train the regression algorithm using the training set.
			this.regression.train(this.samples);
			
			// Store the accuracy for later use.
			this.accuracy = this.regression.accuracy(samples);
		}
		catch (Throwable e) {
			throw new RankingException("Unable to train underling regression algorithm: "+e.getMessage(), e);
		}
	}

	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("PairwiseRanking(");
		buf.append(this.regression.toString());
		buf.append(")");
		return buf.toString();
	}

	/**
	 * Returns the name of the algorithm.
	 * @return the name of the algorithm.
	 */
	public String name() {
		return "PairwiseRanking("+this.regression.name()+")";
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
		try {
			final double aval = this.regression.test(subject.getFeatures().append(a.getFeatures()));
			final double bval = this.regression.test(subject.getFeatures().append(b.getFeatures()));
			if (aval > bval)
				return -1.0;
			else
				return 1.0;
		}
		catch (Throwable e) {
			throw new RankingException("Unable to determine ranking order: "+e.getMessage(), e);
		}
	}
	
	private class RankingObjectSorter implements Comparable<RankingObjectSorter> {
		
		private RankingObject object;
		private double score;
		
		private RankingObjectSorter(RankingSubject subject, RankingObject object) throws RankingException {
			try {
				this.score = regression.test(subject.getFeatures().append(object.getFeatures()));
				this.object = object;
			}
			catch (RegressionException e) {
				throw new RankingException("Unable to determine ranking object score: "+e.getMessage(), e);
			}
		}

		public int compareTo(RankingObjectSorter other) {
			if (this.score > other.score)
				return -1;
			else if (this.score > other.score)
				return 1;
			else
				return 0;
		}
		
		public RankingObject getRankingObject() {
			return this.object;
		}

	}


	public void sort(RankingSubject subject, List<RankingObject> objects) throws RankingException {
		final List<RankingObjectSorter> wrappers = new ArrayList<RankingObjectSorter>();
		for (RankingObject object : objects) {
			wrappers.add(new RankingObjectSorter(subject, object));
		}
		
		Collections.sort(wrappers);
		
		objects.clear();
		
		for (RankingObjectSorter wrapper : wrappers) {
			objects.add(wrapper.getRankingObject());
		}
	}
}
