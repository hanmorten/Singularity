package org.singularity.algorithms;

import org.apache.commons.math3.linear.*;

/**
 * Encapsulates a single training sample, including its feature vector
 * and assigned label (output). The contents of this object is essentially:
 * <pre>
 *   X = [ x1, x2, ... , xn ]
 *   Y = { -1 , +1 }
 * </pre>
 */
public class TrainingSample implements java.io.Serializable {

	/** UUID for serialization. */
	private static final long serialVersionUID = 544968346283579322L;

	/** Feature vector of this training sample. */
	private RealVector features;

	/** Label (output) for this training sample. */
	private double label;
	
	/** Labels (output) for this training sample. */
	private RealVector labels;
	
	/** Weight assigned to this sample (used for some algorithms). */
	private double weight = 1.0d;
	
	/**
	 * Creates a new training sample.
	 * @param features Feature vector of training sample.
	 * @param label Label/output for training sample.
	 */
	public TrainingSample(RealVector features, double label) {
		this.features = features;
		this.label = label;
	}
	
	/**
	 * Creates a new training sample.
	 * @param features Feature vector of training sample.
	 * @param label Label/output for training sample.
	 */
	public TrainingSample(double[] features, double label) {
		this.features = new ArrayRealVector(features);
		this.label = label;
	}

	/**
	 * Creates a new training sample.
	 * @param features Feature vector of training sample.
	 * @param labels Labels/output for training sample.
	 */
	public TrainingSample(RealVector features, RealVector labels) {
		this.features = features;
		this.labels = labels;
		this.label = labels.getEntry(0);
	}

	/**
	 * Creates a new training sample.
	 * @param features Feature vector of training sample.
	 * @param labels Labels/output for training sample.
	 */
	public TrainingSample(double[] features, double[] labels) {
		this.features = new ArrayRealVector(features);
		this.labels = new ArrayRealVector(labels);
		this.label = labels[0];
	}

	/**
	 * Returns the feature vector for the training sample.
	 * @return the feature vector for the training sample.
	 */
	public RealVector getFeatures() {
		return this.features;
	}
	
	/**
	 * Returns the label/output for the training sample.
	 * @return the label/output for the training sample.
	 */
	public double getLabel() {
		return this.label;
	}

	/**
	 * Sets/updates the label for this training sample.
	 * @param label Label for this training sample.
	 */
	public void setLabel(double label) {
		this.label = label;
	}
	
	/**
	 * Returns the labels/output for the training sample.
	 * @return the labels/output for the training sample.
	 */
	public RealVector getLabels() {
		return this.labels;
	}

	/**
	 * Sets/updates the labels for this training sample.
	 * @param labels Labels for this training sample.
	 */
	public void setLabels(RealVector labels) {
		this.labels = labels;
	}

	/**
	 * Returns the weight assigned to this training sample.
	 * @return the weight assigned to this training sample.
	 */
	public double getWeight() {
		return this.weight;
	}
	
	/**
	 * Assigns a new weight to this training sample.
	 * @param weight New weight to assign to this training sample.
	 */
	public void setWeight(double weight) {
		this.weight = weight;
	}
	
	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("TrainingSample(x={");
		for (int i=0; i<features.getDimension(); i++) {
			if (i > 0) buf.append(',');
			buf.append(' ');
			buf.append(features.getEntry(i));
		}
		buf.append(" }, y=");
		if (this.labels == null) {
			buf.append(this.label);
		}
		else {
			buf.append("{");
			for (int i=0; i<labels.getDimension(); i++) {
				if (i > 0) buf.append(',');
				buf.append(' ');
				buf.append(labels.getEntry(i));
			}
			buf.append("}");
		}
		buf.append(", w=");
		buf.append(this.weight);
		buf.append(')');
		return buf.toString();
	}
	
	public boolean equals(Object other) {
		return features.equals(((TrainingSample)other).features);
	}
	
	public int hashCode() {
		return features.hashCode();
	}
	
	
	
}
