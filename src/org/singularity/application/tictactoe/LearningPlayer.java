package org.singularity.application.tictactoe;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.linear.RealVector;
import org.singularity.algorithms.LearningAlgorithm;
import org.singularity.algorithms.LearningListener;
import org.singularity.algorithms.TrainingSample;
import org.singularity.algorithms.TrainingSet;

/**
 * Tic-tac-toe player that optimises its moves using a machine learning
 * algorithm.
 */
public abstract class LearningPlayer extends MachinePlayer {

	/** Number of previous moves to remember and use for training. */
	private final static int MEMORY = 15000;

	/** File to load/store training samples in. */
	private String filename;

	/** Set of training samples from past games. */
	private TrainingSet set = null;

	/** List of feature vectors that represent the current game. */
	protected List<RealVector> features = new ArrayList<RealVector>();

	/**
	 * This callback interface provides information during training of the
	 * learning algorithm we'll use.
	 */
	protected class PlayerLearningListener implements LearningListener {

		/** Player the learning is run for. */
		private Player player;

		/**
		 * Creates a new learning listener.
		 */
		public PlayerLearningListener(Player player) {
			this.player = player;
		}

		/**
		 * Indicates that training is starting.
		 * @param algorithm Learning algorithm instance.
		 */
		public void trainingStart(LearningAlgorithm algorithm) {
			System.out.println(this.player+" starting learning process...");
		}

		/**
		 * Callback that is invoked for each training iteration. Note that some
		 * learning algorithms do not have a fixed number of iterations, and for
		 * these algorithms the 'total' parameter will be zero.
		 * @param algorithm Learning algorithm instance.
		 * @param iteration Iteration number.
		 * @param total Total number of iterations projected (or 0 if total is
		 *    unknown).
		 */
		public void trainingIteration(LearningAlgorithm algorithm, int iteration, int total) {
			if (iteration == 0 && total > 0) {
				for (int i=0; i<total; i++) {
					System.out.print("=");
				}
				System.out.println("");
			}
			System.out.print("#");
		}

		/**
		 * Indicates that training is complete.
		 * @param algorithm Learning algorithm instance.
		 */
		public void trainingEnd(LearningAlgorithm algorithm, TrainingSet samples) {
			System.out.println("");
			try {
				System.out.println("Accuracy: "+algorithm.accuracy(samples));
			}
			catch (Throwable e) {
				e.printStackTrace(System.err);
			}
			System.out.println(this.player+" completed training and is ready to play!");
		}
	}

	/**
	 * Creates a new machine player.
	 * @param name Player name.
	 * @param filename File to load/store training data in.
	 */
	public LearningPlayer(String name, String filename, LearningAlgorithm algorithm) {
		super(name);
		this.filename = filename;
		this.set = this.load();
		//this.set.reduceTo(MEMORY);
		algorithm.setLearningListener(new PlayerLearningListener(this));

		// Update the samples so that each board state is stored with a
		// score that represents the best possible outcome of a game.
		final Map<RealVector,Double> filtered = new HashMap<RealVector,Double>();
		for (int i=0; i<this.set.size(); i++) {
			final TrainingSample sample = this.set.get(i);
			final RealVector features = sample.getFeatures();
			final double score = sample.getLabel();
			final Double current = filtered.get(features);
			if (current == null) {
				filtered.put(features, new Double(score));
			}
			else if (current.doubleValue() < score) {
				filtered.put(features, new Double(score));
			}
		}

		this.set = new TrainingSet();
		for (RealVector features : filtered.keySet()) {
			this.set.add(features, filtered.get(features).doubleValue());
		}

		this.train(this.set, algorithm);
		//System.exit(-1);
		//this.printBadMoves(this.set);
	}

	private void printBadMoves(TrainingSet samples) {
		int bad = 0;
		for (int j=0; j<samples.size(); j++) {
			final TrainingSample sample = samples.get(j);
			if (sample.getLabel() > 0) continue;

			final double[] features = sample.getFeatures().toArray();
			System.out.println("\n\n");
			for (int i=0; i<9; i++) {
				if (i % 3 == 0) {
					System.out.println("+---+---+---+");
				}
				System.out.print("|");
				if (features[i+9] == +1) {
					System.out.print(" ? ");
				}
				else if (features[i] == +1) {
					System.out.print(" X ");
				}
				else if (features[i] == -1) {
					System.out.print(" O ");
				}
				else {
					System.out.print("   ");
				}
				if (i % 3 == 2) 
					System.out.println("|");
			}
			System.out.println("+---+---+---+");
			bad++;
		}
		
		System.err.println("bad="+bad+", tot="+samples.size());
	}

	/**
	 * Tells the extending class to train its learning algorithm.
	 * @param samples Set of training samples.
	 * @param algorithm Learning algorithm to use for training.
	 */
	protected abstract void train(TrainingSet samples, LearningAlgorithm algorithm);

	/**
	 * Triggers start of a new game.
	 */
	public void startGame(Game game) {
		super.startGame(game);
		this.features.clear();
	}

	/**
	 * Determines the next best move based on a learning algorithm.
	 * @param game Game being played.
	 * @return Next move to make (0 to 8).
	 */
	protected abstract int getTrainedMove(Game game);

	/**
	 * Returns this player's next move. The move is as much as possible
	 * based on the learning algorithm. If the learning algorithm can't
	 * provide a next best move, then a random move is made.
	 * @param game Reference to game that is in play.
	 * @return player's next move.
	 */
	public int getNextMove(Game game) {
		int moves = 0;
		final int[] board = game.getBoard();
		for (int i=0; i<board.length; i++) {
			if (board[i] != Game.EMPTY) moves++;
		}

		this.started = (moves % 2) == 0; 

		int move = -1;

		// Attempt to find next best move using learning algorithm.
		try {
			move = this.getTrainedMove(game);
		}
		catch (Throwable e) {
			System.err.println("Error making trained move: "+e.getMessage());
			move = -1;
		}

		// If no best move could be found, a random one is made.
		if (move < 0) {
			move = this.getRandomMove(game);
			System.out.println(move2String(move)+" (random)");
			this.stats.random();
		}
		else {
			System.out.println(move2String(move)+" (educated)");
			this.stats.educated();
		}

		this.myMoves.add(new Integer(move));
		this.features.add(this.getFeatureVector(game, move));
		return move;
	}

	/**
	 * This method tells the extending class to construct a feature vector
	 * representing the next move made by this player. The structure of this
	 * feature vector is completely controlled by the extending class.
	 * @param game The game being played.
	 * @param move The next move made.
	 * @return feature vector representing move made.
	 */
	protected abstract RealVector getFeatureVector(Game game, int move);

	/**
	 * Tells the player that he has won the current game.
	 * @param game The game that was won.
	 */
	public void won(Game game) {
		super.won(game);
		this.storeTrainingSet(this.getTrainingSamples(this.features, +1.0));
	}

	/**
	 * Tells the player that he drew the current game.
	 * @param game The game where the player drew.
	 */
	public void drew(Game game) {
		super.drew(game);
		this.storeTrainingSet(this.getTrainingSamples(this.features, 0.0));
	}

	/**
	 * Tells the player that he has lost the current game.
	 * @param game The game that was lost.
	 */
	public void lost(Game game) {
		super.lost(game);
		this.storeTrainingSet(this.getTrainingSamples(this.features, -1.0));
	}

	/**
	 * Creates a set of training samples based on the outcome of the current
	 * game. These training samples will later be used to train this player.
	 * @param features List of feature vectors that represent our moves in
	 *   the game (as returned by #getFeatureVector(Game,int).
	 * @param outcome Game outcome, +1 for having won the game, -1 for having
	 *   lost the game, and 0 for a draw.
	 * @return list of training samples that represent our moves in the game.
	 */
	protected abstract TrainingSet getTrainingSamples(List<RealVector> features, double outcome);

	/**
	 * Stores the training set in a file.
	 * @param samples Training set
	 */
	protected void storeTrainingSet(TrainingSet samples) {
		//this.set.reduceTo(MEMORY);
		this.set.add(samples);
		this.store(this.set);
	}

	/**
	 * Loads a training set from a file.
	 * @return Training set loaded from file.
	 */
	public TrainingSet load() {
		InputStream istr = null;
		ObjectInputStream oistr = null;

		try {
			final File file = new File(filename);
			istr = new FileInputStream(file);
			oistr = new ObjectInputStream(istr);
			return (TrainingSet)oistr.readObject();
		}
		catch (Throwable e) {
			System.err.println("Error reading file "+filename+": "+e.getMessage());
			return new TrainingSet();
		}
		finally {
			try {
				oistr.close();
			}
			catch (Throwable e) {
				// ignore
			}
			try {
				istr.close();
			}
			catch (Throwable e) {
				// ignore
			}
		}
	}

	/**
	 * Stores a training set in a file.
	 * @param samples Training set.
	 */
	public void store(TrainingSet samples) {
		OutputStream ostr = null;
		ObjectOutputStream oostr = null;

		try {
			final File file = new File(this.filename);
			ostr = new FileOutputStream(file);
			oostr = new ObjectOutputStream(ostr);
			oostr.writeObject(samples);
			oostr.flush();
			ostr.flush();
		}
		catch (Throwable e) {
			System.err.println("Error reading file "+filename+": "+e.getMessage());
		}
		finally {
			try {
				oostr.close();
			}
			catch (Throwable e) {
				// ignore
			}
			try {
				ostr.close();
			}
			catch (Throwable e) {
				// ignore
			}
		}
	}

	/**
	 * Returns statistics on the behaviour and success of this player.
	 */
	public String getStats() {
		return this.stats.toString();
	}

}
