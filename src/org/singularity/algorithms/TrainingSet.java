package org.singularity.algorithms;

import java.io.*;
import java.util.*;

import org.apache.commons.math3.linear.*;

/**
 * Encapsulates a set of training samples.     
 */
public class TrainingSet implements java.io.Serializable {

	/** UUID for serialization. */
	private static final long serialVersionUID = 293213193141283363L;

	/** List of training samples. */
	private List<TrainingSample> samples = new ArrayList<TrainingSample>();
	
	/**
	 * Creates a new empty training set.
	 */
	public TrainingSet() {
		
	}

	/**
	 * Creates a new training setand loads content from a file.
	 * @param file File to read training samples from.
	 * @param limit Maximum number of training samples to read.
	 */
	public TrainingSet(File file, int limit) throws IOException {
		this.parse(file, limit);
	}

	/**
	 * Creates a new training set.
	 * @param samples Training samples to initialize set with.
	 */
	public TrainingSet(List<TrainingSample> samples) {
		this.samples = samples;
	}
	
	/**
	 * Adds a new training sample to the set.
	 * @param sample Training sample.
	 */
	public void add(TrainingSample sample) {
		this.samples.add(sample);
	}

	/**
	 * Adds new training samples to the set.
	 * @param samples Training samples to add.
	 */
	public void add(TrainingSet samples) {
		this.samples.addAll(samples.samples);
	}

	/**
	 * Adds a new training sample to the set.
	 * @param features Feature vector of training sample.
	 * @param label Label/output of training sample.
	 */
	public void add(RealVector features, double label) {
		this.add(new TrainingSample(features, label));
	}
	
	/**
	 * Adds a new training sample to the set.
	 * @param features Feature vector of training sample.
	 * @param label Label/output of training sample.
	 */
	public void add(double[] features, double label) {
		this.add(new TrainingSample(new ArrayRealVector(features), label));
	}

	/**
	 * Returns all training samples in the set.
	 * @return all training samples in the set.
	 */
	public List<TrainingSample> getSamples() {
		return this.samples;
	}
	
	/**
	 * Returns the size of the feature vector of the training samples.
	 * @return the size of the feature vector of the training samples.
	 */
	public int getFeatureVectorSize() {
		if (this.samples.size() == 0) return 0;
		return this.samples.get(0).getFeatures().getDimension();
	}
	
	/**
	 * Returns the size of the training set.
	 * @return the size of the training set.
	 */
	public int size() {
		return this.samples.size();
	}
	 
	/**
	 * Reduces the training set down to a certain size, keeping only the
	 * most recent training samples.
	 * @param size Target training set size.
	 */
	public void reduceTo(int size) {
		while (this.size() > size) {
			this.samples.remove(0);
		}
	}
	
	/**
	 * Returns a single training sample from the set..
	 * @param pos Position/index of training sample.
	 * @return Training sample at position 'pos'.
	 */
	public TrainingSample get(int pos) {
		return this.samples.get(pos);
	}
	
	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("TraningSample (\n");
		for (int i=0; i<this.size(); i++) {
			buf.append((i+1));
			buf.append(" = ");
			buf.append(this.samples.get(i));
			buf.append("\n");
		}
		buf.append(")");
		return buf.toString();
	}

	/**
	 * Reads training samples from a file.
	 * @param file File to read training samples from.
	 * @param limit Maximum number of training samples to read.
	 */
	private void parse(File file, int limit) throws IOException {
		final List<Map<String,Double>> list = new ArrayList<Map<String,Double>>();
		final BufferedReader reader = new BufferedReader(new FileReader(file));
		String line;
		int highest = 0;
		while ((line = reader.readLine()) != null) {
			final Map<String,Double> map = new HashMap<String,Double>();
			String[] tokens = line.split(" ");
			map.put("LABEL", new Double(tokens[0]));
			for (int i=1; i<tokens.length; i++) {
				String[] pair = tokens[i].split(":");
				final String index = pair[0];
				final int parsed = Integer.parseInt(index);
				if (parsed > highest) highest = parsed;
				map.put(index, new Double(pair[1]));  
			}
			list.add(map);
		}
		
		for (int i=0; i<list.size() && i<limit; i++) {
			final Map<String,Double> map = list.get(i);
			final double[] features = new double[highest];
			final double label = map.get("LABEL").doubleValue();
			for (int j=0; j<highest; j++) {
				Double value = map.get("" + j);
				if (value != null)
					features[j] = value.doubleValue();
				else
					features[j] = 0;
			}
			this.add(features, label);
		}
		reader.close();
	}
}
