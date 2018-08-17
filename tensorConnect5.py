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
from tensorflow import keras

gamma = 0.99
split_size = 10
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
	certainty_percentile -- will choose the move in the top n% of predictions.
	"""
	result = []
	data = 0
	game = connect5.GameBoard()
	result.append([[],[],[]])
	result.append([[],[],[]])
	maximum = {"score":0.0}
	# percent, timeout = 0, 0

	while not game.game_over:
		# print(percent)
		# percent += 1
		game._print_board()
		maximum["score"]=0.0

		for i in range(225):
			try:
				copy = game.copy()
				copy.make_move(i//15,i%15)
				board = np.array([copy.gameboard])
				board = board.reshape(board.shape[0], 15, 15, 1)

				predictions = model1.predict(board)[0]
				# print(predictions)
				if predictions[0] > maximum["score"]:
					maximum["score"] = predictions[0]
					maximum["board"] = copy
					maximum["move"] = (i//15,i%15)

				

			except connect5.InvalidMoveError:
				pass

		result[data][1].append(0)
		game = maximum["board"]
		result[data][0].append(np.array([game.gameboard]) / 2)

		# print(coord, game.score_move(row, col))
		game.flip_board()
		# game._switch_turn()
		data = switch_data(data)
		# game._print_board() 

	if game.game_over:
		result[data][1][-1] = 100
		# print("SOMEONE WONNNNN at ", row, col)

	result[0][1] = discount(result[0][1], gamma, True)
	result[1][1] = discount(result[1][1], gamma, True)

	# print(result[0][1])
	# print(result[0][2])
	# print(result[1][1])
	# print(result[1][2])

	# game._print_board()


	return result

def generate_training_set(model1, num_elements):
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
	results = []
	percent_done = 0
	for _ in range(num_elements):
		print("Generating... {}%\r".format(percent_done))
		percent_done += 100//num_elements
		results.extend(generate_training_info(model1, 85))

	ret = max(results, key= lambda x: sum(x[1]))

	ret[0] = np.array(ret[0])
	ret[0] = ret[0].reshape(ret[0].shape[0], 15, 15, 1)
	ret[1] = np.array(ret[1])
	

	return ret

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
			data = generate_training_set(model1, 1)
			print("Generation", gen)

			model1.fit(data[0], data[1], epochs=5)
			gen += 1

	except KeyboardInterrupt:
		print("Exiting, saving model...")
		model1.save(checkpoint_dir)
		print("Saved!")

		data = generate_training_set(model1, 100)
		print("so far,")
		temp = (data[0][-1])
		ref = {0.0: "-", 0.5: "X", 1.0:"O"}
		for thing in temp:
			for piece in [x[0] for x in thing]:
				print(ref[piece],end=" ")
			print()







