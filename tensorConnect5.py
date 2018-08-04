import tensorflow as tf
import numpy as np
import connect5
import random
import os
from tensorflow import keras

gamma = 0.75
checkpoint_path = "connect_5/connect5_model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)


def create_model():
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

		# keras.layers.Flatten(),
		keras.layers.Dense(256, activation = tf.nn.relu),
		keras.layers.Dropout(0.6),

		keras.layers.Dense(512, activation = tf.nn.relu),
		keras.layers.Dropout(0.6),

		keras.layers.Dense(512, activation = tf.nn.relu),
		keras.layers.Dropout(0.6),

		keras.layers.Dense(225, activation = tf.nn.softmax)
	])

	model.compile(optimizer = keras.optimizers.Adam(),
		loss = "sparse_categorical_crossentropy",
		metrics = ["accuracy"])

	return model

def generate_training_info(model1, certainty_percentile):
	result = []
	data = 0
	game = connect5.GameBoard()
	# game.training_baord()
	env = []
	scores = []
	actions = []
	result.append([[],[],[]])
	result.append([[],[],[]])
	timeout = 0

	while not game.game_over and timeout < 6:
		# game._print_board()

		board = np.array([game.gameboard])
		board = board.reshape(board.shape[0], 15, 15, 1)
		predictions = model1.predict(board)[0]
		print(predictions.shape)
		# print(predictions)

		choice_limit = (int)(((100-certainty_percentile)/100)*224) + 1
		coord = random.randrange(choice_limit)
		coord = predictions.argsort()[::-1][coord]

		row, col = coord//15, coord%15
		# print(row, col)

		result[data][0].append(np.array([game.gameboard]) / 2)
		result[data][1].append(coord)
		result[data][2].append(game.score_move(row, col))

		# print(coord, game.score_move(row, col))

		try:

			game.make_move(row, col)
			game.flip_board()
			# game._switch_turn()
			data = switch_data(data)
			# game._print_board() 
			timeout = 0
		except connect5.InvalidMoveError:
			timeout += 1
			continue

	if game.game_over:
		result[switch_data(data)][2][-1] = -100
		# print("SOMEONE WONNNNN at ", row, col)

	result[0][2] = discount(result[0][2], gamma, False)
	result[1][2] = discount(result[1][2], gamma, False)

	# print(result[0][1])
	# print(result[0][2])
	# print(result[1][1])
	# print(result[1][2])

	# game._print_board()


	return result

def generate_training_set(model1, num_elements):
	results = []
	for x in range(num_elements):
		results.extend(generate_training_info(model1, 85))

	ret = max(results, key= lambda x: sum(x[2]))

	ret[0] = np.array(ret[0])
	ret[0] = ret[0].reshape(ret[0].shape[0], 15, 15, 1)
	ret[1] = np.array(ret[1])
	

	return ret

def switch_data(current):
	return 1 if current==0 else 0

def discount(r, gamma, normal):
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
			data = generate_training_set(model1, 100)
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







