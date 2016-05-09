package org.singularity.application.tictactoe;

import java.util.*;

import org.apache.commons.math3.linear.*;
import org.singularity.algorithms.LearningAlgorithm;
import org.singularity.algorithms.TrainingSet;
import org.singularity.algorithms.classifier.Classifier;

/**
 * Tic-tac-toe machine player that learns using a classifier algorithm.
 */
public class ClassifierPlayer extends LearningPlayer {

	/** Classifier algorithm to use for training. */
	private Classifier classifier;
	
	/**
	 * Creates a new machine player.
	 * @param filename File to load/store training data in.
	 * @param classifier Classifier algorithm to use for training.
	 */
	public ClassifierPlayer(String filename, Classifier classifier) {
		super(classifier.getName(), filename, classifier);
	}

	/**
	 * Tells the extending class to train its learning algorithm.
	 * @param samples Set of training samples.
	 * @param algorithm Learning algorithm to use for training.
	 */
	protected void train(TrainingSet samples, LearningAlgorithm algorithm) {
		
		if (samples.size() > 10) {
			try {
				this.classifier = (Classifier)algorithm;
				System.out.println(this+" training based on previous games ("+samples.size()+" moves)...");
				this.classifier.train(samples);
			}
			catch (Throwable e) {
				System.err.println("Unable to train classifier: "+e.getMessage());
				e.printStackTrace(System.err);
				this.classifier = null;
			}
		}
	}

	/**
	 * Determines the next best move based on a learning algorithm.
	 * @param game Game being played.
	 * @return Next move to make (0 to 8).
	 */
	protected int getTrainedMove(Game game) {
		if (this.classifier == null) return -1;
		
		final List<Integer> moves = new ArrayList<Integer>();
		final boolean[] allowed = game.getAllowedMoves();

		/*
		final int[] board = game.getBoard();
		System.out.println("\n\n");
		for (int i=0; i<allowed.length; i++) {
			if (i % 3 == 0) {
				System.out.println("+-------+-------+-------+");
			}
			System.out.print("|");
			if (allowed[i]) {
				try {
					final RealVector features = this.getFeatureVector(game, i);
					final double score = this.classifier.test(features);
					
					final NumberFormat format = NumberFormat.getInstance();
					
					format.setMaximumFractionDigits(3);
					format.setMinimumFractionDigits(3);
					format.setMinimumIntegerDigits(2);
					format.setMaximumIntegerDigits(2);
					
					if (score >= 0) System.out.print("+");
					System.out.print(format.format(score));
					if (score > 0) moves.add(new Integer(i));
				}
				catch (Throwable e) {
					// Ignore
				}
			}
			else if (board[i] == Game.X) {
				System.out.print("   X   ");
			}
			else if (board[i] == Game.O) {
				System.out.print("   O   ");
			}
			else {
				System.out.print("       ");
			}

			if (i % 3 == 2) 
				System.out.println("|");
		}
		System.err.println("+-------+-------+-------+");
	*/
		for (int i=0; i<allowed.length; i++) {
			if (allowed[i]) {
				try {
					final RealVector features = this.getFeatureVector(game, i);
					final double score = this.classifier.test(features);
					if (score > 0) moves.add(new Integer(i));
				}
				catch (Throwable e) {
					e.printStackTrace(System.err);
					System.err.println("Unable to test move: "+e.getMessage());
				}
			}
		}

		if (moves.size() == 0) {
			return -1;
		}
		else if (moves.size() == 1) {
			return moves.get(0).intValue();
		}
		else {
			final int index = new Random().nextInt(moves.size());
			return moves.get(index).intValue();
		}
	}

	/**
	 * This method tells the extending class to construct a feature vector
	 * representing the next move made by this player. The structure of this
	 * feature vector is completely controlled by the extending class.
	 * @param game The game being played.
	 * @param move The next move made.
	 * @return feature vector representing move made.
	 */
	protected RealVector getFeatureVector(Game game, int move) {
		final int[] board = game.getBoard();
		final double[] features = new double[18];

		// Features 0 - 8 represent the state of the board.
		for (int i=0; i<9; i++) {
			if (board[i] == this.mark) {
				features[i] = +1.0d;
			}
			else if (board[i] == Game.EMPTY) {
				features[i] = 0.0d;
			}
			else {
				features[i] = -1.0d;
			}
		}

		features[move + 9] = +1.0d;
		return new ArrayRealVector(features);
	}

	/**
	 * Creates a set of training samples based on the outcome of the current
	 * game. These training samples will later be used to train this player.
	 * @param samples List of feature vectors that represent our moves in
	 *   the game (as returned by #getFeatureVector(Game,int).
	 * @param outcome Game outcome, +1 for having won the game, -1 for having
	 *   lost the game, and 0 for a draw.
	 * @return list of training samples that represent our moves in the game.
	 */
	protected TrainingSet getTrainingSamples(List<RealVector> samples, double outcome) {
		// Convert the game outcome into a training label for the classifier.
		
		// Our outcome is positive if we didn't lose.
		double label = outcome;
		if (label == 0) label = +1.0d;
		
		// Get training samples based on our own moves.
		final TrainingSet set = new TrainingSet();
		for (RealVector features : samples) {
			set.add(features, label);
		}
		
		// Create training samples for the opponent's moves, so that we can
		// learn from the other player's mistakes and cleverness.
		
		label = outcome * -1.0;
		if (label == 0) label = +1.0d;
		final double[] features = new double[18];

		if (!started) {
			// Other player started the game.
			for (int i=0; i<otherMoves.size(); i++) {
				final int move = otherMoves.get(i);
				features[move + 9] = +1.0;
				set.add(features, label);
				features[move + 9] = 0.0;
				features[move] = +1.0;
				if (myMoves.size() > i)
					features[myMoves.get(i)] = -1.0d;
			}
		}
		else {
			// We started the game.
			for (int i=0; i<otherMoves.size(); i++) {
				features[myMoves.get(i)] = -1.0d;
				final int move = otherMoves.get(i);
				features[move + 9] = +1.0;
				set.add(features, label);
				features[move + 9] = 0.0;
				features[move] = +1.0;
			}
		}
		
		return set;
	}

}
