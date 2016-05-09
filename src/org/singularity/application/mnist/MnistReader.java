package org.singularity.application.mnist;

import java.io.*;
import java.nio.ByteBuffer;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.*;
import org.singularity.algorithms.classifier.ClassifierException;
import org.singularity.algorithms.neuralnet.NeuralNetwork;
import org.singularity.algorithms.neuralnet.NeuralNetworkConfiguration;

public class MnistReader {

	private InputStream images;
	private InputStream labels;
	
	private long count;
	private long read = 0;
	private long rows;
	private long cols;
	private long size;
	
	public MnistReader(String images, String labels) throws Exception {
		
		try {
			this.images = new FileInputStream(images);
			this.images = new BufferedInputStream(this.images);
		}
		catch (Throwable e) {
			throw new Exception("Unable to open mnist images file: "+e.getMessage(), e);
		}
		
		try {
			this.labels = new FileInputStream(labels);
			this.labels = new BufferedInputStream(this.labels);
		}
		catch (Throwable e) {
			throw new Exception("Unable to open mnist labels file: "+e.getMessage(), e);
		}

		this.parseHeaders();
		
		this.size = this.rows * this.cols;
	}
	
	public int getCount() {
		return (int)this.count;
	}
	
	public int getImageSize() {
		return (int)this.size;
	}
	
	private long toLong(byte[] bytes) {
		return ByteBuffer.wrap(bytes).getLong();
	}
	
	private void parseHeaders() throws Exception {
		final byte[] buffer = new byte[8];

		try {
			this.labels.read(buffer, 4, 4);
			final long magic = toLong(buffer);
			if (magic != 2049)
				throw new Exception("MNIST labels file header is invalid: "+magic);
			
			this.labels.read(buffer, 4, 4);
			this.count = toLong(buffer);
		}
		catch (Throwable e) {
			throw new Exception("Error parsing MIST labels file: "+e.getMessage(), e);
		}

		try {
			this.images.read(buffer, 4, 4);
			final long magic = toLong(buffer);
			if (magic != 2051)
				throw new Exception("MNIST images file header is invalid: "+magic);
			
			this.images.read(buffer, 4, 4);
			if (this.count != toLong(buffer))
				throw new Exception("Labels file and images file do not correspond! Label file has "+this.count+" entries and images file has "+toLong(buffer)+" entries.");

			this.images.read(buffer, 4, 4);
			this.rows = toLong(buffer);
			this.images.read(buffer, 4, 4);
			this.cols = toLong(buffer);
		}
		catch (Throwable e) {
			throw new Exception("Error parsing MNIST labels file: "+e.getMessage(), e);
		}
	}
	
	private TrainingSample getNextTrainingSample() throws Exception {
		final byte[] label = new byte[1];

		try {
			this.labels.read(label, 0, 1);
		}
		catch (Throwable e) {
			throw new Exception("Error reading MNIST label: "+e.getMessage(), e);
		}

		final byte[] image = new byte[(int)this.size];
		try {
			this.images.read(image, 0, (int)this.size);
		}
		catch (Throwable e) {
			throw new Exception("Error reading MNIST image: "+e.getMessage(), e);
		}

		// Create the output labels
		final RealVector labels = new ArrayRealVector(10);
		labels.setEntry((int)label[0], 1.0d);
		
		// Create the input features (image pixels).
		final RealVector features = new ArrayRealVector((int)this.size);
		for (int i=0; i<this.size; i++) {
			double pixel = (double)image[i];
			pixel = pixel + 128.0d;
			pixel = pixel / 256.0d;
			features.setEntry(i, pixel);
		}
		
		this.read++;
		
		return new TrainingSample(features, labels);
	}

	public TrainingSet getTrainingSet(int max) throws Exception {
		final TrainingSet samples = new TrainingSet();
		for (int i=0; i<max && read<count; i++) {
			samples.add(getNextTrainingSample());
		}
		return samples;
	}

	public void close() {
		try {
			this.images.close();
		}
		catch (Throwable e) {
			// ignore
		}
		try {
			this.labels.close();
		}
		catch (Throwable e) {
			// ignore
		}
	}
	
	/**
	 * Computes the accuracy of the classifier (post training).
	 * @param samples Training samples with assigned labels.
	 * @return Accuracy on a scale from 0.0 to 1.0.
	 * @throws ClassifierException on any testing error.
	 */
	public double accuracy(NeuralNetwork classifier, TrainingSet samples) throws ClassifierException {
		if (samples.size() == 0) return 0.0d;
		double correct = 0;
		for (int i=0; i<samples.size(); i++) {
			final TrainingSample sample = samples.get(i);
			final RealVector result = classifier.testMultiple(sample.getFeatures());
			double highest = -1.0d;
			int number = -1;
			for (int j=0; j<result.getDimension(); j++) {
				double have = result.getEntry(j);
				if (have > highest) {
					highest = have;
					number = j;
				}
			}
			for (int j=0; j<result.getDimension(); j++) {
				result.setEntry(j, 0.0d);
			}
			result.setEntry(number, 1.0d);
			if (result.equals(sample.getLabels())) correct++;
		}
		return correct / (double)samples.size();
	}

	public static void main(String[] args) {
		try {
			if (args.length != 2)
				throw new Exception("Usage: "+MnistReader.class.getName()+" <mnist-images-file> <mnist-labels-file>");

			final MnistReader mnist = new MnistReader(args[0], args[1]);
			final TrainingSet input = mnist.getTrainingSet(1000);
			System.err.println("Loaded "+input.size()+" MNIST images of size "+mnist.cols+"x"+mnist.rows);
			
			final NeuralNetworkConfiguration config = new NeuralNetworkConfiguration();
			config.setIterations(50);
			config.addLayer(mnist.getImageSize(), true);
			config.addLayer(50, true);
			config.addLayer(10, false);
			
			final NeuralNetwork classifier = new NeuralNetwork(config);
			classifier.setLearningListener(new LearningListenerStdout());

			classifier.train(input);
			System.err.println("Accuracy: "+mnist.accuracy(classifier, input));
		}
		catch (Throwable e) {
			System.err.println("ERROR: "+e.getMessage());
			e.printStackTrace(System.err);
		}
	}
}
