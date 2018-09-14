"""
Author Vinh Truong

Convolution Neural Network to play games against itself and train based on
the data produced.
Works with my implementation of connect5.py
"""
import tensorflow as tf
import numpy as np
import connect5
import random
import sys
import os
from math import log2
from heap import maxHeap
from tensorflow import keras
from data_manipulation import generate_batch_equiv

generations = 500

class Agent(object):

	def __init__(self, gamma:float = .75, epsilon:float = 0.8, learning:float = 0.002):
		
		self.gamma = gamma
		self.board_size = 225
		self.epsilon = epsilon
		self.learning_rate = learning
		self.generation = 0
		self.model = self.create_model()
		checkpoint_path = "connect_5/connect5_model.h5"
		self.checkpoint_dir = os.path.dirname(checkpoint_path)
	
	def _increment_generation(self):
		"""
		Automatically updates the epsilon to be 1/(log2(generation))
		"""
		if self.epsilon > 0.1:
			self.epsilon = 1/log2(self.generation+2)
		self.generation += 1

	def train(self):
		# the epsilon is generated by max of 1/generation and the min (0.10)
		data = self.generate_training_set(10)
		print("Generation", self.generation)
		self.model.fit(data[0], data[1], epochs=1)
		self._increment_generation()

	def create_model(self):
		"""
		Creates a keras CNN model. Several convolution and maxpooling layers,
		followed by densely connected layers to a softmax output layer
		"""

		model = keras.Sequential([
			keras.layers.Conv2D(input_shape=(15,15,1), filters=1024, 
								kernel_size=(5,5), padding="same",
								data_format="channels_last"),

			keras.layers.MaxPool2D(input_shape=(1,15,15,1024),pool_size=3,
									strides=None),

			keras.layers.Conv2D(input_shape=(5,5,1024), filters=256,
								kernel_size=(3,3), padding="same",
								data_format="channels_last"),

			keras.layers.Conv2D(input_shape=(5,5,256), filters=128,
								kernel_size=(3,3), padding="same",
								data_format="channels_last"),

			keras.layers.Flatten(),
			keras.layers.Dense(256, activation = tf.nn.relu),
			# keras.layers.Dropout(0.6),

			keras.layers.Dense(512, activation = tf.nn.relu),
			# keras.layers.Dropout(0.6),

			keras.layers.Dense(512, activation = tf.nn.relu),
			# keras.layers.Dropout(0.6),

			keras.layers.Dense(225, activation = tf.nn.softmax)
		])

		model.compile(optimizer = keras.optimizers.Adam(lr=self.learning_rate),
			loss = self.customLoss)

		return model

	def generate_training_info(self, verbose=False):
		"""
		Makes the model play a full game against itself. The game stops
		if the model either wins or makes 5 invalid moves in a row

		Keywords arguments:
		model1 -- The model to predict the next move
		"""
		result = []
		data = 0
		game = connect5.GameBoard()
		result.append([[],[], None])
		result.append([[],[], None])

		while not game.game_over:
			# Prints out the board for user observation
			if verbose:
				if data == 0:
					sys.stdout.write((str(game)))
					sys.stdout.flush()
				if data == 1:
					output = game.copy()
					output.flip_board()
					sys.stdout.write((str(output)))
					sys.stdout.flush()
			
			board = np.array([game.copy().gameboard])
			board = board.reshape(board.shape[0], 15, 15, 1)

			predictions = self.model.predict(board)[0]
			action = [0]*len(predictions)
			predictions = list(zip(range(self.board_size), predictions))
			# predictions is a list of tuples, [(move, prediction_score), ...]
				
			predictions.sort(key=lambda x: x[1], reverse=True)
			move = predictions[random.randint(0, int(self.epsilon*self.board_size)-1)][0]
			# Picks a random move within the epsilon
			action[move] = 1
			# only affects the weight of the current move

			try:
				game_copy = game.copy()
				game_copy.make_move(move//15, move%15)

				result[data][1].append(action)
				result[data][0].append(np.array([game.gameboard]))
				# result[data] contains the results of generating games, 
				# result[data][0] are the envs, result[data][1] are the moves
				# result[data][2] contains one elem, True if won, else False

				game = game_copy

				game.flip_board()
				data = self.switch_data(data)
			except connect5.InvalidMoveError:
				# print("Invalid Move")
				pass

		if game.game_over and game.game_over != -1:
			result[data][2] = True
			result[self.switch_data(data)][2] = False

		# result[0][1] = discount(result[0][1], gamma, True)
		# result[1][1] = discount(result[1][1], gamma, True)

		return result

	def generate_training_set(self, num_elements, verbose=False):
		"""
		Generates a set of data, returns a list of 3 lists.
		The first includes board state data for each move
		The second includes the moves taken by each player
		The third includes the scores for each move (discounted)

		NOTE: there are two of these sets per game, so even though
		one model plays a game by itself, it generates two sets, one
		for each player

		Keyword Arguments:
		model1 -- the model to play the games
		num_elements -- the number of games to play
		"""
		percent_done = 0
		train, target = [], []

		temp = (self.generate_training_info(verbose))

		if temp[0][2] == True:
			train = temp[0][0]
			target = temp[0][1]
		elif temp[1][2] == True:
			# Need to elif in case that there is no winner
			train = temp[1][0]
			target = temp[1][1]

		print("Generating... {}%\r".format((percent_done*100)//num_elements))
		for _ in range(num_elements-1):
			percent_done += 1
			# Information for the user
			temp = (self.generate_training_info(verbose))

			print(np.argmax(target), np.argmin(target))
			
			if temp[0][2] == True and len(temp[0][1]) < len(target):
				train = temp[0][0]
				target = temp[0][1]
			elif temp[1][2] == True and len(temp[1][1]) < len(target):
				# Need to elif in case that there is no winner
				train = temp[1][0]
				target = temp[1][1]
			print("Generating... {}%\r".format((percent_done*100)//num_elements))
		
		train, target = generate_batch_equiv(train, target)

		train = np.array(train)
		train = train.reshape(train.shape[0], 15, 15, 1)
		target = np.array(target)
		# print(target)

		return (train, target)

	def save_model(self):
		self.model.save(self.checkpoint_dir)
		print("Saved!")


	def switch_data(self, current):
		"""
		used in generate_training_info to split up data
		by player
		"""
		return 1 if current==0 else 0

	def discount(self, r, gamma, normal):
		"""
		will discount the moves so that moves that lead
		to good moves will get some extra points
		"""
		discount = np.zeros_like(r)
		G = 0.0
		for i in reversed(range(0, len(r))):
			G = G * gamma + r[i]
			discount[i] = G
		# Normalize 
		if normal:
			mean = np.mean(discount)
			std = np.std(discount)
			discount = (discount - mean) / (std)
		return discount

	def customLoss(self, target, pred):
		"""
		Returns a mean squared error loss
		"""
		loss = (target-pred)**2
		return loss
	

if __name__ == "__main__":
	agent = Agent()
	try:
		while True:
			agent.train()
			# agent.save_model()

	except KeyboardInterrupt:
		print("Exiting, saving model...")

		agent.save_model()