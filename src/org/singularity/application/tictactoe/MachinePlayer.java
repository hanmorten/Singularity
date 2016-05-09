package org.singularity.application.tictactoe;

import java.io.*;
import java.util.*;

import org.apache.commons.math3.linear.*;
import org.singularity.*;

/**
 * A machine tic-tac-toe player, trained using a learning algorithm of some
 * shape or form.
 */
public abstract class MachinePlayer extends Player {

	/** Game moves made by this player. */
	protected List<Integer> myMoves = new ArrayList<Integer>();
	/** Game moves made by opponent. */
	protected List<Integer> otherMoves = new ArrayList<Integer>();

	/** Game statistics container. */
	protected Statistics stats = new Statistics();
	
	/** True if this player made the first move in the current game. */
	protected boolean started = false;

	/**
	 * Container for game statistics.
	 */
	protected class Statistics {
		
		/** Number of random moves made. */
		private int random = 0;
		/** Number of educated moves made. */
		private int educated = 0;
		/** Number of games won. */
		private int wins = 0;
		/** Number of games lost. */
		private int losses = 0;
		/** Number of games drawn. */
		private int draws = 0;
	
		/** Creates a new statistics container. */
		public Statistics() {
			
		}
	
		/** Signals that a random move was made. */
		public void random() {
			this.random++;
		}
		
		/** Signals that a educated move was made. */
		public void educated() {
			this.educated++;
		}
		
		/** Signals that a game was won. */
		public void won() {
			this.wins++;
		}
		
		/** Signals that a game was lost. */
		public void lost() {
			this.losses++;
		}
		
		/** Signals that a game was drewn. */
		public void drew() {
			this.draws++;
		}
		
		/*
		 * (non-Javadoc)
		 * @see java.lang.Object#toString()
		 */
		public String toString() {
			final StringBuffer buf = new StringBuffer();
			buf.append("Games=");
			buf.append(this.wins + this.draws + this.losses);
			buf.append(" (Wins=");
			buf.append(this.wins);
			buf.append(", Draws=");
			buf.append(this.draws);
			buf.append(", Losses=");
			buf.append(this.losses);
			buf.append("), Moves=");
			buf.append(this.educated + this.random);
			buf.append(" (Educated=");
			buf.append(this.educated);
			buf.append(", Random=");
			buf.append(this.random);
			buf.append(")");
			return buf.toString();
		}

	}
	
	/**
	 * Creates a new machine player.
	 * @param name Player name.
	 */
	public MachinePlayer(String name) {
		super(name);
	}

	/**
	 * Triggers start of a new game.
	 */
	public void startGame(Game game) {
		this.myMoves.clear();
		this.otherMoves.clear();
	}

	/**
	 * Determines the next best move based on a random selection.
	 * @param game Game being played.
	 * @return Next move to make (0 to 8).
	 */
	protected int getRandomMove(Game game) {
		final boolean[] allowed = game.getAllowedMoves();
		int empty = 0;
		for (int i=0; i<allowed.length; i++) {
			if (allowed[i]) {
				empty++;
			}
		}

		if (empty == 0) {
			return -1;
		}
		else if (empty == 1) {
			for (int i=0; i<allowed.length; i++) {
				if (allowed[i]) {
					return i;
				}
			}
		}
		else {
			int index = new Random().nextInt(empty);
			for (int i=0; i<allowed.length; i++) {
				if (allowed[i]) {
					if (index == 0) {
						return i;
					}
					index--;
				}
			}
		}
		
		return -1;
	}
	
	/**
	 * Returns this player's next move. The move is as much as possible
	 * based on the learning algorithm. If the learning algorithm can't
	 * provide a next best move, then a random move is made.
	 * @param game Reference to game that is in play.
	 * @return player's next move.
	 */
	public abstract int getNextMove(Game game);

	/**
	 * Tells this player that the other player made a move.
	 * @param game Game where the other player made a move.
	 * @param move The move that the other player made.
	 */
	public void otherPlayerMoved(Game game, int move) {
		otherMoves.add(new Integer(move));
	}

	/**
	 * Tells the player that he has won the current game.
	 * @param game The game that was won.
	 */
	public void won(Game game) {
		this.stats.won();
	}
	
	/**
	 * Tells the player that he drew the current game.
	 * @param game The game where the player drew.
	 */
	public void drew(Game game) {
		this.stats.drew();
	}

	/**
	 * Tells the player that he has lost the current game.
	 * @param game The game that was lost.
	 */
	public void lost(Game game) {
		this.stats.lost();
	}
	
	/**
	 * Returns statistics on the behaviour and success of this player.
	 */
	public String getStats() {
		return this.stats.toString();
	}
	
}
