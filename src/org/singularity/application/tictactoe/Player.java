package org.singularity.application.tictactoe;

/**
 * Base class for all tic-tac-toe players (machine or human).
 */
public abstract class Player {

	/** Player name (normal human readable name). */
	private String name;
	
	/** The mark this player uses (Game.X or Game.O). */
	protected int mark = 0;
	
	/**
	 * Creates a new player with a given name.
	 * @param name Player name.
	 */
	public Player(String name) {
		this.name = name;
	}

	/**
	 * Sets the mark this player should use.
	 * @param mark Mark this player should use (Game.X or Game.O).
	 */
	public void setMark(int mark) {
		this.mark = mark;
	}
	
	/**
	 * Translates a board index (0 to 8) to a move ID (such as A2).
	 * @param move Board index for move.
	 * @return Move identifier, such as C3.
	 */
	protected String move2String(int move) {
		final String[] cols = { "A","B","C" };
		return cols[move % 3] + (move / 3 + 1);
	}
	
	/**
	 * Returns a description of this player.
	 * @return a description of this player.
	 */
	public String toString() {
		return this.name;
	}

	/**
	 * Tells the player that a new game is about to start.
	 * @param game Game that is about to start.
	 */
	public void startGame(Game game) {
		
	}
	
	/**
	 * Returns this player's next move.
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
		
	}

	/**
	 * Tells the player that he has won the current game.
	 * @param game The game that was won.
	 */
	public void won(Game game) {
		System.out.println(this+" won");
	}

	/**
	 * Tells the player that he has lost the current game.
	 * @param game The game that was lost.
	 */
	public void lost(Game game) {
		System.out.println(this+" lost");
	}
	
	/**
	 * Tells the player that he drew the game.
	 * @param game The game that the player drew.
	 */
	public void drew(Game game) {
		System.out.println(this+" drew");
	}
	
	/**
	 * Returns some statistics information for this player.
	 * @return some statistics information for this player.
	 */
	public String getStats() {
		return "";
	}
	
}
