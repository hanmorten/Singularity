package org.singularity.application.tictactoe;

import java.util.*;
import java.text.*;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.singularity.*;
import org.singularity.algorithms.LearningAlgorithm;
import org.singularity.algorithms.TrainingSample;
import org.singularity.algorithms.TrainingSet;
import org.singularity.algorithms.regression.Regression;

/**
 * Tic-tac-toe machine player that learns using a classifier algorithm.
 */
public class RegressionPlayer extends LearningPlayer {

	private Regression regression;
	
	public RegressionPlayer(String filename, Regression regression) {
		super(regression.getName(), filename, regression);
	}

	/**
	 * Tells the extending class to train its learning algorithm.
	 * @param samples Set of training samples.
	 * @param algorithm Learning algorithm to use for training.
	 */
	protected void train(TrainingSet samples, LearningAlgorithm algorithm) {
		
		if (samples.size() > 10) {
			try {
				this.regression = (Regression)algorithm;
				System.out.println(this+" training based on previous games ("+samples.size()+" moves)...");
				this.regression.train(samples);
			}
			catch (Throwable e) {
				System.err.println("Unable to train regression algorithm: "+e.getMessage());
				e.printStackTrace(System.err);
				this.regression = null;
			}
		}
	}

	/**
	 * Determines the next best move based on a learning algorithm.
	 * @param game Game being played.
	 * @return Next move to make (0 to 8).
	 */
	protected int getTrainedMove(Game game) {
		if (this.regression == null) return -1;
		
		int bestMove = -1;
		double bestScore = -10000;
		final int[] board = game.getBoard();
		final boolean[] allowed = game.getAllowedMoves();
		
		/*
		System.out.println("\n\n");
		for (int i=0; i<allowed.length; i++) {
			if (i % 3 == 0) 
				System.out.println("+-------+-------+-------+");
			System.out.print("|");
			if (allowed[i]) {
				try {
					final RealVector features = this.getFeatureVector(game, i);
					final double score = this.regression.test(features);
					
					final NumberFormat format = NumberFormat.getInstance();
					
					format.setMaximumFractionDigits(4);
					format.setMinimumFractionDigits(4);
					format.setMinimumIntegerDigits(2);
					format.setMaximumIntegerDigits(2);
					
					System.out.print(format.format(score));
				}
				catch (Throwable e) {
					// ignore
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
					final double score = this.regression.test(features);
					if (score > bestScore) {
						bestScore = score;
						bestMove = i;
					}
				}
				catch (Throwable e) {
					e.printStackTrace(System.err);
					System.err.println("Unable to test move: "+e.getMessage());
				}
			}
		}

		return bestMove;
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

		// Features 9 - 17 represent the next move.
		features[move+9] = +1.0d;
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
		final TrainingSet set = new TrainingSet();
		
		double label = outcome;
		for (RealVector vector : samples) {
			set.add(new TrainingSample(vector, label));
		}

		// Create training samples for the opponent's moves, so that we can
		// learn from the other player's mistakes and cleverness.
		
		label = outcome * -1.0;
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
