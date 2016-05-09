package org.singularity.application.tictactoe;

/**
 * A tic-tac-toe player that makes random (but legal) moves.
 */
public class RandomPlayer extends MachinePlayer {

	/**
	 * Creates a new random player.
	 */
	public RandomPlayer() {
		super("Random");
	}

	/*
	 * (non-Javadoc)
	 * @see org.singularity.application.tictactoe.MachinePlayer#getNextMove(org.singularity.application.tictactoe.Game)
	 */
	public int getNextMove(Game game) {
		this.stats.random();
		return super.getRandomMove(game);
	}

}
