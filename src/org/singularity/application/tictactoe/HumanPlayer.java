package org.singularity.application.tictactoe;

import java.util.Scanner;

/**
 * Human player for tic-tac-toe game. This player expects moves to be entered 
 * from the terminal/stdin, as A1 for top-left corner and C3 for bottom right
 * corner of the 3x3 board.
 */
public class HumanPlayer extends Player {

	/**
	 * Creates a new human player.
	 */
	public HumanPlayer() {
		super("Human");
	}
	
	/*
	 * (non-Javadoc)
	 * @see org.singularity.application.tictactoe.Player#getNextMove(org.singularity.application.tictactoe.Game)
	 */
	public int getNextMove(Game game) {
		
		// Repeat until a valid move has been entered.
		while (1 == 1) {
			Scanner in = null;
			try {
				// Get next line input.
				in = new Scanner(System.in);
				final String str = in.next().toUpperCase();
				if (str.length() != 2) throw new Exception("Invalid input "+str);
					
				int col = 0;
				if (str.contains("A"))
					col = 0;
				else if (str.contains("B"))
					col = 1;
				else if (str.contains("C"))
					col = 2;
				else
					throw new Exception("Invalid input "+str);

				int row = 0;
				if (str.contains("1"))
					row = 0;
				else if (str.contains("2"))
					row = 1;
				else if (str.contains("3"))
					row = 2;
				else
					throw new Exception("Invalid input "+str);
				
				return row * 3 + col;
			}
			catch (Throwable e) {
				//e.printStackTrace(System.err);
				//System.err.println("Error: "+e.getMessage());
			}
			finally {
				try {
					//in.close();
				}
				catch (Throwable e) {
					// ignore.
				}
			}
		}
	}

}
