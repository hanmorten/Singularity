package org.singularity.algorithms.neuralnet;

/**
 * Synapse that connects one neuron to another. These synapses are
 * used to pass output values from one neuron as a weighted input
 * value to another neuron. Note that weights are applied when data
 * passes over the synapse, and not within the origin neuron. This
 * allows the output from one neuron to be weighted differently
 * when passed as input to different neurons.
 */
public class Synapse {

	/**
	 * Weights assigned to synapses leading into a Neuron. These weights are
	 * set and used by the perceptrons within the neurons to produce their
	 * overall input value.
	 */
	public class Weight {

		private double value;
		private double change;
		
		/**
		 * Creates a new weight with a random value between -0.5 and 0.5.
		 */
		public Weight() {
			this.value = Math.random() - 0.5d;
			this.change = 0.0d;
		}

		/**
		 * Creates a new weight with a given value.
		 * @param weight The value to be assigned to the weight.
		 */
		public Weight(double weight) {
			this.value = weight;
			this.change = 0.0d;
		}

		/**
		 * Increments the weight by a given amount.
		 * @param amount The amount to increment the weight by.
		 */
		public void increment(double amount) {
		    value += amount;
		}

		/**
		 * Decrements the weight by a given amount.
		 * @param amount The amount to decrement the weight by.
		 */
		public void decrement(double amount) {
		    value -= amount;
		}

		/**
		 * Sets the value of the weight.
		 * @param weight The value to assign to the weight.
		 */
		public void set(double weight) {
		    value = weight;
		}

		/**
		 * Returns the value of the weight.
		 * @return The value of the weight.
		 */
		public double get() {
		    return value;
		}

		/**
		 * Sets the delta/change value of the weight.
		 * @param change The change value of the weight.
		 */
		public void setChange(double change) {
		    this.change += change;
		}

		/**
		 * Returns the delta/change value of the weight.
		 * @return The delta/change value of the weight.
		 */
		public double getChange() {
		    return change;
		}

	}
	
	/** Input neuron for this synapse. */
	private Neuron input;
	/** Output neuron for this synapse. */
	private Neuron output;
	/** Weight applies to this synapse. */
	private Weight weight;
	
	/**
	 * Creates a new synapse between two neurons.
	 * @param from Input neuron.
	 * @param to Ouptut neuron.
	 */
	public Synapse(Neuron from, Neuron to) {
		this.input = from;
		this.output = to;
		this.weight = new Weight();
	}
	
	/**
	 * Returns the weight for the synapse.
	 * @return The weight for the synapse.
	 */
	public Weight getWeight() {
	    return weight;
	}

	/**
	 * Returns the input neuron that sends data over this synapse.
	 * @return This synapse's input neuron.
	 */
	public Neuron getInputNeuron() {
	    return input;
	}

	/**
	 * Returns the output neuron that receives data over this synapse.
	 * @return This synapse's output neuron.
	 */
	public Neuron getOutputNeuron()  {
	    return output;
	}

	/**
	 * Returns the input value for this synapse.
	 * @return The input value for this synapse.
	 */
	public double getInput() {
	    return this.input.getOutput();
	}

	/**
	 * Returns the weighted input value for this synapse.
	 * @return The weighted input value for this synapse.
	 */
	public double getWeightedInput() {
	    return this.input.getOutput() * this.weight.get();
	}

}
