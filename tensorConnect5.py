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
from heap import maxHeap
from tensorflow import keras

gamma = 0.75
board_size = 225
epsilon_min = 0.1
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

		keras.layers.Conv2D(input_shape=(5,5,256), filters=128,
							kernel_size=(3,3), padding="same",
							data_format="channels_last"),

		keras.layers.Flatten(),
		keras.layers.Dense(256, activation = tf.nn.relu),
		keras.layers.Dropout(0.6),

		keras.layers.Dense(512, activation = tf.nn.relu),
		keras.layers.Dropout(0.6),

		keras.layers.Dense(512, activation = tf.nn.relu),
		keras.layers.Dropout(0.6),

		keras.layers.Dense(225, activation = tf.nn.softmax)
	])

	model.compile(optimizer = keras.optimizers.Adam(),
		loss = "mean_squared_error",
		metrics = ["accuracy"])

	return model

def generate_training_info(model1, epsilon, verbose=False):
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

		predictions = model1.predict(board)[0]
		action = [0.]*len(predictions)
		predictions = list(zip(range(board_size), predictions))
		# predictions is a list of tuples, [(move, prediction_score), ...]
			
		predictions.sort(key=lambda x: x[1], reverse=True)
		move = predictions[random.randint(0, int(epsilon*board_size)-1)][0]
		# Picks a random move within the epsilon
		action[move] = 1
		# only affects the weight of the current move

		try:
			game.make_move(move//15, move%15)

			result[data][1].append(action)
			result[data][0].append(np.array([game.gameboard]))
			# result[data] contains the results of generating games, 
			# result[data][0] are the envs, result[data][1] are the moves
			# result[data][2] contains one elem, True if won, else False

			game.flip_board()
			data = switch_data(data)
		except connect5.InvalidMoveError:
			# print("Invalid Move")
			pass

	if game.game_over and game.game_over != -1:
		result[data][2] = True
		result[switch_data(data)][2] = False

	# result[0][1] = discount(result[0][1], gamma, True)
	# result[1][1] = discount(result[1][1], gamma, True)

	return result

def generate_training_set(model1, num_elements, epsilon, verbose=False):
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

	temp = (generate_training_info(model1, epsilon, verbose))

	if temp[0][2] == True:
		result[train] = temp[0][0]
		result[target] = temp[0][1]
	elif temp[1][2] == True:
		# Need to elif in case that there is no winner
		result[train] = temp[1][0]
		result[target] = temp[1][1]

	for _ in range(num_elements-1):
		print("Generating... {}%\r".format((percent_done*100)//num_elements))
		percent_done += 1
		# Information for the user

		temp = (generate_training_info(model1, epsilon, verbose))
		
		if temp[0][2] == True and len(temp[0][1]) < len(result[target]):
			result[train] = temp[0][0]
			result[target] = temp[0][1]
		elif temp[1][2] == True and len(temp[1][1]) < len(result[target]):
			# Need to elif in case that there is no winner
			result[train] = temp[1][0]
			result[target] = temp[1][1]

	result[train] = np.array(result[train])
	result[train] = result[train].reshape(result[train].shape[0], 15, 15, 1)
	result[target] = np.array(result[target])
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
	gen = 1
	try:
		while True:
			# the epsilon is generated by max of 1/generation and the min (0.10)
			epsilon = ((1/gen)+1)/2
			epsilon = max(epsilon_min, epsilon)
			data = generate_training_set(model1, 10, epsilon)
			print("Generation", gen-1)
			gen += 1
			model1.fit(data[0], data[1], epochs=2)
			# model1.save(checkpoint_dir)

	except KeyboardInterrupt:
		print("Exiting, saving model...")
		model1.save(checkpoint_dir)
		print("Saved!")

		data = generate_training_set(model1, 5, 0.1, True)
		print("so far,")
		temp = (data[0][-1])
		ref = {0.0: "-", 0.5: "X", 1.0:"O"}
		for thing in temp:
			for piece in [x[0] for x in thing]:
				print(ref[piece],end=" ")
			print()