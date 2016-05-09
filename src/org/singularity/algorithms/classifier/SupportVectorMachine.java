package org.singularity.algorithms.classifier;

import java.io.File;
import java.util.*;

import org.apache.commons.math3.linear.*;
import org.singularity.*;
import org.singularity.algorithms.LearningListenerStdout;
import org.singularity.algorithms.TrainingSample;
import org.singularity.algorithms.TrainingSet;
import org.singularity.algorithms.kernel.*;

/**
 * Implements the Support Vector Machine classifier using the SMO
 * (Sequential Minimal Optimization) algorithm.     
 */
public class SupportVectorMachine extends Classifier implements java.io.Serializable {

	/** UID required for serialization. */
	private static final long serialVersionUID = 3293629989313592039L;

	/** Parameter: Complexity constant. */ 
	private double C = 1.0f;
	/** Parameter: Error tolerance. */ 
	private double tolerance = 0.00001;
	/** Parameter: Epsilon (alpha variance tolerance). */ 
	private double epsilon = 0.00001;
	/** Parameter: Number of times to iterate over the alpha's without changing. */
	private int iterations = 10;
	/** Parameter: Kernel function to use. */
	private Kernel kernel;

	/** Reduced set of training samples that we need to retain for testing. */
	private TrainingSet samples;

	/** Output: The alpha values to apply during testing. */
	private double[] alpha = new double[0];
	/** Output: The bias or offset. */
	private double bias;

	/** List of support vectors. */
	private List<RealVector> supportVectors = new ArrayList<RealVector>();

	/**
	 * Creates a new support vector machine classifier.
	 * @param kernel The kernel function to use with the SVM.
	 */
	public SupportVectorMachine(Kernel kernel) {
		this.kernel = kernel;
	}

	/**
	 * Creates a new support vector machine classifier.
	 * @param kernel The kernel function to use with the SVM.
	 * @param C the complexity constant (normally 1).
	 */
	public SupportVectorMachine(Kernel kernel, double C) {
		this.kernel = kernel;
		this.C = C;
	}

	/**
	 * Creates a new support vector machine classifier.
	 * @param kernel The kernel function to use with the SVM.
	 * @param C the complexity constant (normally 1).
	 * @param tolerance Error tolerance (default 0.001).
	 * @param epsilon Alpha variance tolerance to accept before resetting
	 *     iteration count to zero (default 0.00001).
	 * @param iterations Number of iterations (with no alpha change) to perform.
	 */
	public SupportVectorMachine(Kernel kernel, double C, double tolerance, double epsilon, int iterations) {
		this.kernel = kernel;
		this.C = C;
		this.tolerance = tolerance;
		this.epsilon = epsilon;
		this.iterations = iterations;
	}

	/**
	 * Trains the classifier using a set of training samples.
	 * @param samples Training samples.
	 * @throws ClassifierException on any error.
	 */
	public void train(TrainingSet samples) throws ClassifierException {
		if (this.listener != null) this.listener.trainingStart(this);

		try {
			this.alpha = new double[samples.size()];
			this.bias = 0;

			int pass = 0;

			while (pass < iterations) {
				if (this.listener != null) this.listener.trainingIteration(this, pass, 0);

				int alpha_change = 0;

				for (int i=0; i<samples.size(); i++) {

					final TrainingSample sample1 = samples.get(i);
					final RealVector x1 = sample1.getFeatures();
					final double label1 = sample1.getLabel();

					final double error1 = test(x1, samples) - label1;

					if ((label1 * error1 < -tolerance && this.alpha[i] < C) ||
							(label1 * error1 > tolerance  && this.alpha[i] > 0)) {

						int j = (int)Math.floor(Math.random() * (samples.size() - 1));
						j = (j < i) ? j : (j + 1);

						final TrainingSample sample2 = samples.get(j);
						final RealVector x2 = sample2.getFeatures();
						final double label2 = sample2.getLabel();

						final double error2 = test(x2, samples) - label2;

						final double oldAlpha1 = this.alpha[i];
						final double oldAlpha2 = this.alpha[j];

						// Compute the value for L.
						double L = 0;
						if (label1 != label2)
							L = Math.max(0, -oldAlpha1 + oldAlpha2);
						else
							L = Math.max(0, oldAlpha1 + oldAlpha2 - C);

						// Compute the new value for H.
						double H = 0;
						if (label1 != label2)
							H = Math.min(C, -oldAlpha1 + oldAlpha2 + C);
						else
							H = Math.min(C, oldAlpha1 + oldAlpha2);

						if (L == H) continue;

						// Cache a few kernel calculations...
						final double kernel1_1 = this.kernel.kernel(x1,x1);
						final double kernel1_2 = this.kernel.kernel(x1,x2);
						final double kernel2_2 = this.kernel.kernel(x2,x2);

						final double eta = kernel1_2 + kernel1_2 - kernel1_1 - kernel2_2;
						if (eta >= 0) continue;

						this.alpha[j] = oldAlpha2 - (label2 * (error1 - error2)) / eta;

						if (this.alpha[j] > H)
							this.alpha[j] = H;
						else if (this.alpha[j] < L)
							this.alpha[j] = L;

						if (Math.abs(this.alpha[j] - oldAlpha2) < epsilon) continue;

						this.alpha[i] = oldAlpha1 + label1 * label2 * (oldAlpha2 - this.alpha[j]);

						// Compute the updated bias.
						final double alpha1 = alpha[i];
						final double alpha2 = alpha[j];
						if (0 < alpha1 && alpha1 < C) {
							final double delta1 = label1 * (alpha1 - oldAlpha1);
							final double delta2 = label2 * (alpha2 - oldAlpha2);
							this.bias = this.bias - error1 - delta1 * kernel1_1 - delta2 * kernel1_2;
						}
						else if (0 < alpha2 && alpha2 < C) {
							final double delta1 = label1 * (alpha1 - oldAlpha1);
							final double delta2 = label2 * (alpha2 - oldAlpha2);
							this.bias = this.bias - error2 - delta1 * kernel1_2 - delta2 * kernel2_2;
						}
						else {
							final double delta1 = label1 * (alpha1 - oldAlpha1);
							final double delta2 = label2 * (alpha2 - oldAlpha2);
							final double bias1 = this.bias - error1 - delta1 * kernel1_1 - delta2 * kernel1_2;
							final double bias2 = this.bias - error2 - delta1 * kernel1_2 - delta2 * kernel2_2;
							this.bias = (bias1 + bias2) / 2;
						}

						alpha_change++;
					}
				}

				if (alpha_change == 0) {
					pass++;
					System.err.print('-');
				}
				else {
					pass = 0;
				}
				
			}

			this.reduce(samples);
		}
		catch (Throwable e) {
			throw new ClassifierException("Error training SVM algorithm: "+e.getMessage(), e);
		}
		finally {
			if (this.listener != null) this.listener.trainingEnd(this, samples);
		}
	}

	/**
	 * Reduces the training set we need to keep to contain only those
	 * that are strictly needed for testing.
	 * @param samples All training samples.
	 */
	private void reduce(TrainingSet samples) {
		// Create a new set of those training samples we need to retain.
		this.samples = new TrainingSet();
		for (int i=0; i<samples.size(); i++) {
			try {
				System.err.println("SAMPLE "+samples.get(i)+", TEST="+this.test(samples.get(i).getFeatures())+", ALPHA "+alpha[i]);
			}
			catch (Throwable e) {
				e.printStackTrace();
			}
			if (this.alpha[i] != 0) {
				this.samples.add(samples.get(i));
			}
		}

		// Update the alphas and keep only non-zero values.
		final double[] alpha = new double[this.samples.size()];
		int j=0;
		for (int i=0; i<this.alpha.length; i++) {
			if (this.alpha[i] != 0) {
				alpha[j++] = this.alpha[i] * samples.get(i).getLabel();
			}
		}

		for (int i=0; i<this.alpha.length; i++) {
			if (this.alpha[i] > 0) {
				this.supportVectors.add(samples.get(i).getFeatures());
			}
		}

		this.alpha = alpha;
	}

	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * @param input Feature vector of test sample.
	 * @return -1 or +1.
	 * @throws ClassifierException on any error.
	 */
	private double test(RealVector x, TrainingSet samples) throws KernelException {
		double f = this.bias;
		for (int i=0; i<samples.size(); i++) {
			final TrainingSample sample = samples.get(i);
			if (this.alpha[i] != 0.0d) {
				f += this.alpha[i] * sample.getLabel() * this.kernel.kernel(x, sample.getFeatures());
			}
		}
		return f;
	}

	public double real(RealVector x) throws ClassifierException {
		try {
			double f = this.bias;
			for (int i=0; i<this.samples.size(); i++) {
				f += this.alpha[i] * this.kernel.kernel(x, this.samples.get(i).getFeatures());
			}
			return f;
		}
		catch (KernelException e) {
			throw new ClassifierException("Error testing using SVM algorithm: "+e.getMessage(), e);
		}
	}

	/**
	 * Makes a predicted output based on a feature vector (testing).
	 * @param x Feature vector of test sample.
	 * @return -1 or +1.
	 * @throws ClassifierException on any error.
	 */
	public double test(RealVector x) throws ClassifierException {
		if (this.real(x) < 0)
			return -1.0d;
		else
			return 1.0d;
	}

	/*
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("SupportVectorMachine(\n  Bias=");
		buf.append(this.bias);
		buf.append(",\n  Kernel=");
		buf.append(this.kernel);
		buf.append(",\n  SupportVectors=(");
		boolean first = true;
		for (int i=0; i<this.alpha.length; i++) {
			if (Math.abs(alpha[i]) == 1.0) {
				if (first)
					first = false;
				else
					buf.append(",");
				final RealVector vector = this.samples.get(i).getFeatures();
				buf.append("[");
				for (int j=0; j<vector.getDimension(); j++) {
					if (j > 0) buf.append(",");
					buf.append(vector.getEntry(j));
				}
				buf.append("] = ");
				buf.append(alpha[i]);
			}
		}
		buf.append(")\n)");
		return buf.toString();
	}

	private boolean isSupportVector(RealVector x) {
		for (RealVector sv : this.supportVectors) {
			if (x.equals(sv)) return true;
		}
		return false;
	}

	/**
	 * Very basic plotting of 2-dimensional problem sets.
	 * @throws ClassifierException on testing errors.
	 */
	public void plot2D() throws ClassifierException {
		final double[] data = new double[2];
		System.out.println("    -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9 ");
		for (int x1 = 9; x1>-10; x1--) {
			System.out.print(" ");
			if (x1 >= 0) System.out.print(" ");
			System.out.print(x1);
			System.out.print(" ");
			for (int x2 = -9; x2<10; x2++) {
				data[0] = x1;
				data[1] = x2;
				final RealVector x = new ArrayRealVector(data);
				if (this.isSupportVector(x)) {
					if (this.test(x) < 0)
						System.out.print("[ ]");
					else
						System.out.print("[#]");
				}
				else {
					if (this.test(x) < 0)
						System.out.print("   ");
					else
						System.out.print("###");
				}
			}
			System.out.println();
		}	
	}

	/**
	 * Very basic plotting of 2-dimensional problem sets.
	 * @throws ClassifierException on testing errors.
	 */
	public void plot2D(TrainingSet set) throws ClassifierException {
		final Map<RealVector,Integer> samples = new HashMap<RealVector,Integer>();
		for (int i=0; i<set.size(); i++) {
			final TrainingSample sample = set.get(i);
			samples.put(sample.getFeatures(), new Integer((int)sample.getLabel()));
		}

		final double[] data = new double[2];
		System.out.println("    -9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9 ");
		for (int x1 = 9; x1>-10; x1--) {
			System.out.print(" ");
			if (x1 >= 0) System.out.print(" ");
			System.out.print(x1);
			System.out.print(" ");
			for (int x2 = -9; x2<10; x2++) {
				data[0] = x1;
				data[1] = x2;
				final RealVector x = new ArrayRealVector(data);
				final Integer label = samples.get(x);
				if (label != null) {
					if (this.isSupportVector(x)) {
						if (label.intValue() < 0)
							System.out.print("[-]");
						else
							System.out.print("[+]");
					}
					else {
						if (label.intValue() < 0)
							System.out.print(" - ");
						else
							System.out.print("#+#");
					}
				}
				else {
					if (this.isSupportVector(x)) {
						if (this.test(x) < 0)
							System.out.print("[ ]");
						else
							System.out.print("[#]");
					}
					else {
						if (this.test(x) < 0)
							System.out.print("   ");
						else
							System.out.print("###");
					}
				}
			}
			System.out.println();
		}	
	}
	
	/**
	 * Returns the name of the algorithm.
	 * @return the name of the algorithm.
	 */
	public String name() {
		return "SupportVectorMachine";
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

			input.add(new double[] { 6,7 },  -1);
			input.add(new double[] { 7,7 },  -1);
			input.add(new double[] { 6,6 },  -1);
			input.add(new double[] { 7,6 },  -1);
			input.add(new double[] { 2,4 },  -1);
			input.add(new double[] { 2,6 },  -1);
			input.add(new double[] { 3,4 },  -1);
			input.add(new double[] { 3,6 },  -1);
			input.add(new double[] { 4,4 },  -1);
			input.add(new double[] { 4,6 },  -1);
			input.add(new double[] { 5,4 },  -1);
			input.add(new double[] { 5,6 },  -1);

			input.add(new double[] { -6,-7 },  -1);
			input.add(new double[] { -7,-7 },  -1);
			input.add(new double[] { -6,-6 },  -1);
			input.add(new double[] { -7,-6 },  -1);
			input.add(new double[] { -2,-4 },  -1);
			input.add(new double[] { -2,-6 },  -1);
			input.add(new double[] { -3,-4 },  -1);
			input.add(new double[] { -3,-6 },  -1);
			input.add(new double[] { -4,-4 },  -1);
			input.add(new double[] { -4,-6 },  -1);
			input.add(new double[] { -5,-4 },  -1);
			input.add(new double[] { -5,-6 },  -1);

			final SupportVectorMachine classifier = new SupportVectorMachine(new PolynomialKernel(0.3d, 0.2d, 3d));
			//final SupportVectorMachine classifier = new SupportVectorMachine(new RadialBasisFunctionKernel(0.5d));
			//final SupportVectorMachine classifier = new SupportVectorMachine(new SigmoidKernel(0.75, 1d));
			//final SupportVectorMachine classifier = new SupportVectorMachine(new LinearKernel());
			classifier.setLearningListener(new LearningListenerStdout());
			classifier.train(input);
			System.err.println("Accuracy: "+classifier.accuracy(input));
			classifier.plot2D(input);

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