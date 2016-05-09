package org.singularity.algorithms.ranking;

import org.apache.commons.math3.linear.*;

/**
 * Encapsulates a subject whose preference we're ranking based on.
 * The subject can be a person, such as a consumer or traveler.     
 */
public interface RankingSubject {

	/**
	 * Returns the feature vector of the ranking subject.
	 * @return the feature vector of the ranking subject.
	 */
	public RealVector getFeatures();
	
}
