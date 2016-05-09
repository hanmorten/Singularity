package org.singularity.application.tictactoe;

import java.util.*;

/**
 * Tic-tac-toe player that always plays the perfect moves. This player is
 * hard-coded (not learning) to play the perfect strategy, as outlined here:
 * https://en.wikipedia.org/wiki/Tic-tac-toe#Strategy
 */
public class PerfectPlayer extends MachinePlayer {

	public PerfectPlayer() {
		super("Perfect");
	}

	public int getNextMove(Game game) {
		this.stats.educated();
		
		final int[] board = game.getBoard();
		final boolean[] allowed = game.getAllowedMoves();
		
		final int ourMark = this.mark;
		final int hisMark = this.mark == Game.X ? Game.O : Game.X;
		
		int move = -1;
		
		List<Integer> moves = null;
		
		// Win: If the player has two in a row, they can place a third to
		// get three in a row.
		if ((moves = getWinningMove(board, ourMark)) != null)
			return this.getRandomMove(moves);

		// Block: If the opponent has two in a row, the player must play
		// the third themselves to block the opponent.
		if ((moves = getWinningMove(board, hisMark)) != null)
			return this.getRandomMove(moves);
		
		// Fork: Create an opportunity where the player has two threats
		// to win (two non-blocked lines of 2).
		if ((moves = getForkMove(board, ourMark)) != null)
			return this.getRandomMove(moves);
		
		// Blocking an opponent's fork:
		// Option 1: The player should create two in a row to force the
		// opponent into defending, as long as it doesn't result in
		// them creating a fork. For example, if "X" has a corner, "O"
		// has the center, and "X" has the opposite corner as well, "O"
		// must not play a corner in order to win. (Playing a corner in
		// this scenario creates a fork for "X" to win.)
		if ((moves = getOffensiveMove(board, ourMark)) != null)
			return this.getRandomMove(moves);
		
		// Option 2: If there is a configuration where the opponent can fork,
		// the player should block that fork.
		if ((moves = getForkMove(board, hisMark)) != null)
			return this.getRandomMove(moves);
		
		// Center: A player marks the center. (If it is the first move of
		// the game, playing on a corner gives "O" more opportunities to
		// make a mistake and may therefore be the better choice; however,
		// it makes no difference between perfect players.)
		if ((move = getCenterMove(allowed)) > -1)
			return move;
		
		// Opposite corner: If the opponent is in the corner, the player
		// plays the opposite corner.
		if ((moves = getOppositeCornerMove(board, hisMark)) != null)
			return this.getRandomMove(moves);
		
		// Empty corner: The player plays in a corner square.
		if ((moves = getCornerMove(allowed)) != null)
			return this.getRandomMove(moves);
		
		// Empty side: The player plays in a middle square on any of the 4 sides.
		if ((moves = getSideMove(board)) != null)
			return this.getRandomMove(moves);
		
		return -1;
	}

	private Integer canPlay(int[] board, int mark, int pos1, int pos2, int pos3) {
		if (board[pos1] == mark && board[pos2] == mark && board[pos3] == Game.EMPTY)
			return new Integer(pos3);
		if (board[pos1] == mark && board[pos3] == mark && board[pos2] == Game.EMPTY)
			return new Integer(pos2);
		if (board[pos2] == mark && board[pos3] == mark && board[pos1] == Game.EMPTY)
			return new Integer(pos1);
		return null;
	}
	
	private int getRandomMove(List<Integer> moves) {
		final int random = new Random().nextInt(moves.size());
		return moves.get(random).intValue();
	}
	
	private List<Integer> getWinningMove(int[] board, int mark) {
		final List<Integer> moves = new ArrayList<Integer>();
		Integer move;
		if ((move = canPlay(board, mark, 0, 1, 2)) != null) moves.add(move);
		if ((move = canPlay(board, mark, 3, 4, 5)) != null) moves.add(move);
		if ((move = canPlay(board, mark, 6, 7, 8)) != null) moves.add(move);
		if ((move = canPlay(board, mark, 0, 3, 6)) != null) moves.add(move);
		if ((move = canPlay(board, mark, 1, 4, 7)) != null) moves.add(move);
		if ((move = canPlay(board, mark, 2, 5, 8)) != null) moves.add(move);
		if ((move = canPlay(board, mark, 0, 4, 8)) != null) moves.add(move);
		if ((move = canPlay(board, mark, 2, 4, 6)) != null) moves.add(move);
		if (moves.size() == 0) return null;
		return moves;
	}

	private int countWins(int[] board, int mark) {
		int wins = 0;
		if (canPlay(board, mark, 0, 1, 2) != null) wins++;
		if (canPlay(board, mark, 3, 4, 5) != null) wins++;
		if (canPlay(board, mark, 6, 7, 8) != null) wins++;
		if (canPlay(board, mark, 0, 3, 6) != null) wins++;
		if (canPlay(board, mark, 1, 4, 7) != null) wins++;
		if (canPlay(board, mark, 2, 5, 8) != null) wins++;
		if (canPlay(board, mark, 0, 4, 8) != null) wins++;
		if (canPlay(board, mark, 2, 4, 6) != null) wins++;
		return wins;
	}

	private List<Integer> getForkMove(int[] board, int mark) {
		final List<Integer> moves = new ArrayList<Integer>();
		final int[] future = new int[9];
		for (int i=0; i<9; i++) {
			if (board[i] == Game.EMPTY) {
				System.arraycopy(board, 0, future, 0, 9);
				future[i] = mark;
				if (this.countWins(future, mark) > 2) {
					moves.add(new Integer(i));
				}
			}
		}
		
		if (moves.size() == 0) return null;
		return moves;
	}

	private List<Integer> getOffensiveMove(int[] board, int mark) {
		final List<Integer> moves = new ArrayList<Integer>();
		final int[] future = new int[9];
		for (int i=0; i<9; i++) {
			if (board[i] == Game.EMPTY) {
				System.arraycopy(board, 0, future, 0, 9);
				future[i] = mark;
				if (this.getWinningMove(future, mark) != null) {
					moves.add(new Integer(i));
				}
			}
		}
		
		if (moves.size() == 0) return null;
		return moves;
	}

	private int getCenterMove(boolean[] allowed) {
		if (allowed[4])
			return 4;
		else
			return -1;
	}

	private List<Integer> getOppositeCornerMove(int[] board, int mark) {
		final List<Integer> moves = new ArrayList<Integer>();
		if (board[0] == Game.EMPTY && board[8] == mark) moves.add(new Integer(0));
		if (board[8] == Game.EMPTY && board[0] == mark) moves.add(new Integer(8));
		if (board[2] == Game.EMPTY && board[6] == mark) moves.add(new Integer(2));
		if (board[6] == Game.EMPTY && board[2] == mark) moves.add(new Integer(6));
		if (moves.size() == 0) return null;
		return moves;
	}

	private List<Integer> getCornerMove(boolean[] allowed) {
		final List<Integer> moves = new ArrayList<Integer>();
		if (allowed[0]) moves.add(new Integer(0));
		if (allowed[2]) moves.add(new Integer(2));
		if (allowed[8]) moves.add(new Integer(8));
		if (allowed[6]) moves.add(new Integer(6));
		if (moves.size() == 0) return null;
		return moves;
	}

	private List<Integer> getSideMove(int[] board) {
		final List<Integer> moves = new ArrayList<Integer>();
		if (board[1] == Game.EMPTY) moves.add(new Integer(1));
		if (board[3] == Game.EMPTY) moves.add(new Integer(3));
		if (board[5] == Game.EMPTY) moves.add(new Integer(5));
		if (board[7] == Game.EMPTY) moves.add(new Integer(7));
		if (moves.size() == 0) return null;
		return moves;
	}

}
