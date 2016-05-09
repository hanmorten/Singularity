package org.singularity.algorithms.ranking;

import org.apache.commons.math3.linear.*;

/**
 * Interface used to encapsulate objects to be ranked. An object can be a
 * product, such as a movie or hotel accommodation.  
 */
public interface RankingObject {

	/**
	 * Returns the feature vector of the ranking object.
	 * @return the feature vector of the ranking object.
	 */
	public RealVector getFeatures();
	
}
