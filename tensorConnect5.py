"""
Author Vinh Truong

Convolution Neural Network to play games against itself and train based on
the data produced.
Works with my implementation of connect5.py
Changed from classification model to a regression model to score and choose the
best move
"""
import tensorflow as tf
import numpy as np
import connect5
import random
import os
from heap import maxHeap
from tensorflow import keras

gamma = 0.99
board_size = 64
checkpoint_path = "connect_5/connect5_model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)


def create_model():
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

		# keras.layers.Conv2D(input_shape=(5,5,256), filters=128,
		# 					kernel_size=(3,3), padding="same",
		# 					data_format="channels_last"),

		keras.layers.Flatten(),
		keras.layers.Dense(256, activation = tf.nn.relu),
		keras.layers.Dropout(0.6),

		keras.layers.Dense(512, activation = tf.nn.relu),
		keras.layers.Dropout(0.6),

		keras.layers.Dense(512, activation = tf.nn.relu),
		keras.layers.Dropout(0.6),

		keras.layers.Dense(1, activation = "sigmoid")
	])

	model.compile(optimizer = keras.optimizers.Adam(),
		loss = "mean_squared_error",
		metrics = ["accuracy"])

	return model

def generate_training_info(model1, certainty_percentile):
	"""
	Makes the model play a full game against itself. The game stops
	if the model either wins or makes 5 invalid moves in a row

	Keywords arguments:
	model1 -- The model to predict the next move
	"""
	result = []
	data = 0
	game = connect5.GameBoard()
	result.append([[],[]])
	result.append([[],[]])

	while not game.game_over:
		game._print_board()
		moves = []
		next_move = None

		for i in range(64):
			try:
				copy = game.copy()
				copy.make_move(i//8,i%8)
				board = np.array([copy.gameboard])
				board = board.reshape(board.shape[0], 15, 15, 1)

				predictions = model1.predict(board)[0]
				moves.append((predictions[0], (i//8,i%8)))
				
			except connect5.InvalidMoveError:
				pass
		
		moves = maxHeap(moves, key=lambda x:x[0])
		for _ in range(random.randint(1, int(certainty_percentile*board_size)+1)):
			if len(moves.tree) == 1:
				break
			next_move = moves.pop()[1]
		game.make_move(next_move[0], next_move[1])

		result[data][1].append(0)
		result[data][0].append(np.array([game.gameboard]) / 2)

		game.flip_board()
		data = switch_data(data)

	if game.game_over:
		result[data][1][-1] = 225
		result[switch_data(data)][1][-1] = -50

	result[0][1] = discount(result[0][1], gamma, True)
	result[1][1] = discount(result[1][1], gamma, True)

	return result

def generate_training_set(model1, num_elements, certainty_percentile):
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
	result = [[],[]]
	train, target = 0, 1
	for _ in range(num_elements):
		print("Generating... {}%\r".format(percent_done))
		percent_done += 100//num_elements
		temp = (generate_training_info(model1, certainty_percentile))
		# print(temp[0][1])
		result[train].append(temp[0][0])
		result[train].append(temp[1][0])
		result[target].append(temp[0][1])
		result[target].append(temp[1][1])
	
	max_score = 0
	max_index = 0
	for i in range(len(result[target])):
		if result[target][i][0] > max_score:
			max_score = result[target][i][0]
			max_index = i
	result[train] = np.array(result[train][max_index])
	result[train] = result[train].reshape(result[train].shape[0], 15, 15, 1)
	result[target] = np.array(result[target][max_index])
	# print(result[target])

	return (result[train], result[target])

def switch_data(current):
	"""
	used in generate_training_info to split up data
	by player
	"""
	return 1 if current==0 else 0

def discount(r, gamma, normal):
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

if __name__ == "__main__":
	model1 = create_model()
	gen = 0
	try:
		while True:
			data = generate_training_set(model1, 1, .15)
			print("Generation", gen)

			model1.fit(data[0], data[1], epochs=2)
			gen += 1

	except KeyboardInterrupt:
		print("Exiting, saving model...")
		model1.save(checkpoint_dir)
		print("Saved!")

		data = generate_training_set(model1, 1, 0.1)
		print("so far,")
		temp = (data[0][-1])
		ref = {0.0: "-", 0.5: "X", 1.0:"O"}
		for thing in temp:
			for piece in [x[0] for x in thing]:
				print(ref[piece],end=" ")
			print()
