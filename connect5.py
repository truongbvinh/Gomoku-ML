# author Vinh Truong 2018

import numpy as np

white = 1
black = 2

for_print = {0: "-", 1: "X", 2: "O"}

dim = 15

class GameBoard:


	def __init__(self):
		"""
		Initializes a 15x15 board to play on
		has attributes: gomeboard, game_over, turn
		"""

		self.gameboard = np.array([np.array([0 for y in range(dim)]) for x in range(dim)])
		self.game_over = False
		self.turn = white

	def copy(self):
		"""
		Returns a new copy of the current instance of a GameBoard
		"""

		result = GameBoard()
		result.gameboard = self.gameboard.copy()
		result.game_over = self.game_over
		result.turn = self.turn
		return result

	def make_move(self, row, col):
		"""
		Places a piece of the current player's turn on the board. if the
		game is won by that move, then sets self.game_over to the winning 
		player. if the intended placement is already occupied by a piece, 
		raises an InvalidMoveError

		Keyword arguments:
		row -- the row of intended placement, 0 based index
		col -- the col of intended placement, 0 based index		
		"""

		if (self.gameboard[row][col] != 0):
			raise InvalidMoveError
		self.gameboard[row][col] = self.turn
		self._check_winner()
		self._switch_turn()

	def _switch_turn(self):
		"""
		Switches the attribute self.turn to the other player
		"""

		if (self.turn == white):
			self.turn = black
		else:
			self.turn = white

	def _check_winner(self):
		"""
		Returns the winner of the game. If there is no winner,
		returns None instead.,
		"""
		is_filled = True
		
		for row in range(dim):
			for col in range(dim):
				if self.gameboard[row][col] != 0:
					temp = self._check_piece(row, col)
					if temp != None:
						return temp
				else:
					is_filled = False
		if is_filled:
			self.game_over = -1
		return None

	def _check_piece(self, row, col):
		"""
		Helper function for self._check_winner
		"""
		
		for i in range(0, 2):
			for j in range(-1, 2):
				if (row+i < 0 or row+i > dim or col+j < 0 or col+j > dim or (i == 0 and j == 0)):
					continue
				temp = self._check_dir(row, col, i, j)
				if temp != None:
					return temp
		return None

	def _check_dir(self, row, col, i, j):
		"""
		Helper function for self._check_piece
		"""

		counter = 1
		while counter != 5:
			try:
				if (self.gameboard[row][col] == self.gameboard[row + (i*counter)][col + (j*counter)]):
					counter += 1
				else:
					return None
			except IndexError:
				return None
		self.game_over = self.gameboard[row][col]
		return self.gameboard[row][col]

	def _print_board(self):
		"""
		Prints the board with 0 based index borders
		"""

		counter = 0
		end = {0:" ", 1:"'"}
		print("  0 1 2 3 4 5 6 7 8 9 0'1'2'3'4'")
		for row in self.gameboard:
			print(counter%10, end=end[counter//10])
			for col in row:
				print(for_print[col], end=" ")
			print()
			counter += 1

		if self.game_over == False:
			print("NEXT TURN: {}".format(for_print[self.turn]))
	
	def __str__(self):
		"""
		Returns a string display of the board
		"""
		result = ""
		result += "  0 1 2 3 4 5 6 7 8 9 0'1'2'3'4'\n"
		counter = 0
		for row in self.gameboard:
			result += "{} ".format(counter)
			for col in row:
				result += "{} ".format(for_print[col])
			result += "\n"
			counter += 1
			counter %= 10
		
		return result + "\r"

	def run_game(self):
		"""
		Plays a game with 2 human players, input is ("[row] [col]")
		Plays until there is a winner, then prints winner and board
		"""
		self._print_board()
		while self._check_winner() == None:
			try:
				row, col = [int(coord) for coord in input().split()]
				print(self.score_move(row, col))
				self.make_move(row, col)
				self._print_board()
			except (InvalidMoveError, IndexError):
				print("Invalid; please try again")
			# print(self._must_be_blocked())
		print("WINNER IS:", self._check_winner())

	def training_board(self):
		"""
		Places two pieces on the board in specified locations.
		Intended use is for ML training, placing hardcoded pieces on board
		"""
		self.make_move(7, 6)
		self.make_move(6, 7)

	def flip_board(self):
		"""
		Switches pieces on board (black -> white and white -> black) and
		switches the turn. 

		Intended use if for ML training, making sure the model playes against
		itself but only trains as 1 player
		"""
		for i in range(dim):
			for j in range(dim):
				if self.gameboard[i][j] == 1:
					self.gameboard[i][j] = 4
		self.gameboard = self.gameboard // 2
		self._switch_turn()

	def score_move(self, row, col):
		"""
		Assigns a score for the current player's intended placement.
		Will give extra points of having multiple 

		Does not make the move or change the board
		"""
		temp = self.copy()
		if temp.gameboard[row][col] != 0:
			return -5
		score = 5

		# Checking proximity for similar pieces and creating 3's
		for i in range(-1,2):
			for j in range(-1,2):
				if i == 0 and j == 0:
					continue
				for k in range(1,5):
					try:
						if temp.gameboard[row + (i*k)][col + (j*k)] == temp.turn:
							score += 5
						else:
							break
					except:
						break
				else:
					score += 50
		return score

	def split_board(self):
		"""
		splits up the board into smaller board, using a "filter" size of 5x5,
		and moves the filter over by 1 row and 1 col as to not miss out on any
		patterns. The reason the filter is 5x5 is because we want to catch patterns
		that are 5 or less pieces in length
		"""
		result = []
		irow, icol = 0, 0
		for i in range(100):
			irow = i // 10
			icol = i % 10
			temp = []
			for row in range(5):
				temp.append(np.array(self.gameboard[irow+row][icol:icol+5]))
			temp = np.array(temp)
			if temp.sum() != 0:
				# only add the snippet if it has a piece in it
				result.append(temp)

		# add in empty snippet in case the board has no pieces
		if len(result) == 0:
			result.append(temp)

		return np.array(result)

	def game_state(self, next_row, next_col):
		"""
		Returns a list of 3 boards.
			First board - Current player's pieces
			Second board - Opponent's pieces
			Third board - Empty board except board[next_row][next_col] = 1
		"""
		result = np.zeros((3, dim, dim))
		for row in range(dim):
			for col in range(dim):
				if self.gameboard[row][col] == self.turn:
					result[0][row][col] = 1
				elif self.gameboard[row][col] != 0:
					result[1][row][col] = 1
		
		result[2][next_row][next_col] = 1

		return result



class InvalidMoveError(Exception):
	pass

if __name__ == "__main__":
	instance = GameBoard()
	instance.run_game()

