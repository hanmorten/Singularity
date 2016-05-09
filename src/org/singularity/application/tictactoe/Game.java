package org.singularity.application.tictactoe;

import java.util.*;

import org.singularity.algorithms.classifier.*;
import org.singularity.algorithms.kernel.*;
import org.singularity.algorithms.regression.*;

/**
 * Representation of a tic-tac-toe game, where players can be either machines
 * (software) or humans. 
 */
public class Game {

	/** Board status: A cell is empty. */ 
	public static final int EMPTY = 0;
	/** Board status: X has places a piece in this cell. */
	public static final int X = 1;
	/** Board status: O has places a piece in this cell. */
	public static final int O = 2;
	/** Game status: Game was a draw. */
	public static final int DRAW = 3;

	/**
	 * Square board represented as an array of X (=1) or O (=2).
	 * Empty cells are represented as zeros. Board layout:
	 * <pre>
	 *   0 1 2
	 *   3 4 5
	 *   6 7 8
	 * </pre>
	 */
	public int[] board = new int[9];

	/** Reference to player X (human or machine). */
	private Player playerX = null;
	/** Reference to player O (human or machine). */
	private Player playerO = null;

	/** Number of moves made. */
	private int moves = 0;

	/** Winner of the game. */
	private int winner = 0;

	/** Order of play. */
	private int order = 0;

	/**
	 * Creates a new tic-tac-toe game.
	 */
	public Game() {

	}

	/**
	 * Returns the layout of the game board as an array.
	 * @return the layout of the game board as an array.
	 */
	public int[] getBoard() {
		return this.board;
	}

	/**
	 * Starts playing a new game.
	 * @param x Player using X (human or machine).
	 * @param o Player using O (human or machine).
	 */
	public void play(Player x, Player o) {
		// Tell the two players which pieces they'll use.
		x.setMark(X);
		o.setMark(O);

		// Reset the game status.
		this.playerX = x;
		this.playerO = o;
		this.moves = 0;
		this.winner = 0;
		Arrays.fill(this.board, EMPTY);

		// Randomize the game order (who makes first move).
		this.order = new Random().nextInt(2);

		// Tell the players that we're about to start.
		this.playerX.startGame(this);
		this.playerO.startGame(this);
		
		// Main game iteration loop.
		while ((this.winner = checkWinner()) == 0) {
			System.out.print(this);

			// Find the player to make the next move.
			final Player player = (moves % 2 == this.order) ? x : o;
			final Player other = (moves % 2 != this.order) ? x : o;
			final int mark = (moves % 2 == this.order) ? X : O;
			
			// Tell the player to make the next move.
			final int move = player.getNextMove(this);
			
			// Validate that the move is legal.
			if (!isAllowedMove(move)) {
				System.err.println("Player "+player+" played illegal move "+move);
				continue;
			}
			
			// Update the board with the player's move.
			this.board[move] = mark;
			this.moves++;
			
			// Tell the other player that this player made the move.
			other.otherPlayerMoved(this, move);
		}

		// Inform the players who won/lost/drew.
		if (this.winner == X) {
			this.playerX.won(this);
			this.playerO.lost(this);
		}
		else if (this.winner == O) {
			this.playerX.lost(this);
			this.playerO.won(this);
		}
		else {
			this.playerX.drew(this);
			this.playerO.drew(this);
		}

		System.out.print(this);
	}

	/**
	 * Checks if a move is legal.
	 * @param move Index of board cell the move was made to.
	 * @return true if move is legal.
	 */
	private boolean isAllowedMove(int move) {
		if (move < 0) return false;
		if (move >= this.board.length) return false;
		final boolean[] allowed = this.getAllowedMoves();
		return allowed[move];
	}
	
	/**
	 * Returns all allowed moves given the current state of the game.
	 * The moves are returned as an array of boolean values, indicating
	 * if a piece can be placed in a cell.
	 * @return all allowed moves given the current state of the game.
	 */
	public boolean[] getAllowedMoves() {
		final boolean[] result = new boolean[9];
		Arrays.fill(result, false);
		
		// Only allowed moves are 1, 3, 5 and 7.
		if (this.moves < 2) {
			if (this.board[1] == EMPTY)
				result[1] = true;
			if (this.board[3] == EMPTY)
				result[3] = true;
			if (this.board[5] == EMPTY)
				result[5] = true;
			if (this.board[7] == EMPTY)
				result[7] = true;
		}
		else {
			for (int i=0; i<9; i++) {
				if (this.board[i] == EMPTY) {
					result[i] = true;
				}
			}
		}
		
		return result;
	}

	/**
	 * Checks is a player has won the game.
	 * @return identifier for who won the game (0 if no win/draw).
	 */
	private int checkWinner() {
		if (board[0] != EMPTY && board[0] == board[1] && board[1] == board[2]) return board[0];
		if (board[3] != EMPTY && board[3] == board[4] && board[4] == board[5]) return board[3];
		if (board[6] != EMPTY && board[6] == board[7] && board[7] == board[8]) return board[6];

		if (board[0] != EMPTY && board[0] == board[3] && board[3] == board[6]) return board[0];
		if (board[1] != EMPTY && board[1] == board[4] && board[4] == board[7]) return board[1];
		if (board[2] != EMPTY && board[2] == board[5] && board[5] == board[8]) return board[2];

		if (board[0] != EMPTY && board[0] == board[4] && board[4] == board[8]) return board[0];
		if (board[2] != EMPTY && board[2] == board[4] && board[4] == board[6]) return board[2];

		int empty = 0;
		for (int i=0; i<this.board.length; i++) {
			if (this.board[i] == EMPTY) {
				empty++;
			}
		}

		if (empty == 0) return DRAW;

		return 0;
	}

	/**
	 * Creates a string representation of the current game status.
	 * @return a string representation of the current game status.
	 */
	public String toString() {
		final StringBuffer buf = new StringBuffer();
		buf.append("\nTic-Tac-Toe: X=");
		buf.append(this.playerX);
		buf.append(", O=");
		buf.append(this.playerO);
		buf.append("\n");
		buf.append("     A   B   C\n");
		buf.append("   +---+---+---+\n");
		for (int r=0; r<3; r++) {
			buf.append(" ");
			buf.append((r+1));
			buf.append(" ");
			for (int c=0; c<3; c++) {
				final int value = this.board[r*3 + c]; 
				switch (value) {
				case X:
					buf.append("| X ");
					break;
				case O:
					buf.append("| O ");
					break;
				default:
					buf.append("|   ");
					break;
				}
			}
			buf.append("|\n");
			buf.append("   +---+---+---+\n");
		}

		final int winner = this.checkWinner();
		if (winner == EMPTY) {
			if (this.moves % 2 == this.order)
				buf.append(this.playerX+"'s move: ");
			else
				buf.append(this.playerO+"'s move: ");
		}
		else if (winner == X) {
			buf.append(this.playerX+" won the game\n");
		}
		else if (winner == O) {
			buf.append(this.playerO+" won the game\n");
		}
		else if (winner == DRAW) {
			buf.append("Game is a draw\n");
		}

		return buf.toString();
	}

	/**
	 * Runs game from command-line.
	 * @param args No arguments are needed.
	 */
	public static void main(String[] args) {
		Player playerA = null;
		Player playerB = null;
		
		for (int i=0; i<10000; i++) {
			// Re-train players every 100 iterations
			if (i % 100 == 0) {
				//playerA = new ClassifierPlayer("tictactoeA.bin", new Boosting(50));
				//playerA = new ClassifierPlayer("tictactoeA.bin", new AveragePerceptron(new PolynomialKernel(2.0d, 0.0d, 1d), 100));
				//playerA = new ClassifierPlayer("tictactoeA.bin", new SupportVectorMachine(new PolynomialKernel(0.51d, 0.2d, 1d), 40));
				//playerA = new ClassifierPlayer("tictactoeA.bin", new AveragePerceptron(new RadialBasisFunctionKernel(8d), 50));
				//playerA = new RegressionPlayer("tictactoeA.bin", new PoissonRegression(50));
				playerA = new PerfectPlayer();
				//playerA = new HumanPlayer();
				//playerA = new RandomPlayer();

				//playerB = new ClassifierPlayer("tictactoeB.bin", new Boosting(100));
				//playerB = new RegressionPlayer("tictactoeB.bin", new PoissonRegression(500));
				//playerB = new RegressionPlayer("tictactoeB.bin", new LinearLeastSquaresRegression());
				//playerB = new RegressionPlayer("tictactoeB.bin", new LogisticRegression(50, 0.1));
				//playerB = new ClassifierPlayer("tictactoeB.bin", new AveragePerceptron(new LinearKernel(), 100));
				//playerB = new ClassifierPlayer("tictactoeB.bin", new AveragePerceptron(new PolynomialKernel(0.74d, 0.01d, 1d), 100));
				//playerB = new ClassifierPlayer("tictactoeB.bin", new SupportVectorMachine(new LinearKernel(), 10));
				playerB = new ClassifierPlayer("tictactoeB.bin", new SupportVectorMachine(new PolynomialKernel(0.74d, 0.01d, 1d), 50));
				//playerB = new ClassifierPlayer("tictactoeB.bin", new SupportVectorMachine(new SigmoidKernel(0.01,0.1), 50));
				//playerB = new HumanPlayer();
				//playerB = new PerfectPlayer();
			}
			
			System.out.println("");
			System.out.println("        ----- =========== -----");
			System.out.println("              TIC-TAC-TOE");
			System.out.println("               Game "+i);
			System.out.println("        ----- =========== -----");
			System.out.println("");
			try {
				final Game game = new Game();
				game.play(playerA, playerB);
			}
			catch (Throwable e) {
				System.err.println("Error: "+e.getMessage());
				e.printStackTrace();
			}

			System.out.println(playerA+" status: "+playerA.getStats());
			System.out.println(playerB+" status: "+playerB.getStats());
		}
		
	}

}
