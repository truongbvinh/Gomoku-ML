"""
Author: Vinh Truong
"""

import numpy as np

class InvalidValueError(Exception):
    pass

def extract_game_state(self, row, col):
    """
    Returns a list of 3 boards.
        First board - Current player's pieces
        Second board - Opponent's pieces
        Third board - Last move
    
    The model always plays as White, so white
    will be the 'current player'
    """
    result = np.zeros(3, dim, dim)
    for row in range(dim):
        for col in range(dim):
            if self.gameboard[row][col] == 1:
                result[0][row][col] = 1
            elif self.gameboard[row][col] == 2:
                result[1][row][col] = 1
    
    result[2][row][col] = 1

    return result

def generate_equiv_train(train):
    """

    Returns a list;
        First list contains 4 copies of the training info, 
        each rotated 90 more than the preceding element

        Second list contains 4 copies of target info, each
        matching the rotations of the board
    """
    train_result = []
    for i in range(4):
        train_result.append(np.rot90(train, i, (1, 2)))

    train_result = np.array(train_result)
    return train_result

def generate_batch_equiv(train_list, data_format="channels_first"):
    train_result = []
    if data_format == "channels_first":
        pass
    elif data_format == "channels_last":
        train_list = train_list.reshape(train_list.shape[0], 3, 15, 15)
    else:
        raise InvalidValueError

    for train in train_list:
        temp = generate_equiv_train(train)
        train_result.extend(generate_equiv_train(train))
    
    if data_format == "channels_last":
        train_list = train_list.reshape(train_list.shape[0], 15, 15, 3)
    
    return np.array(train_result)

if __name__ == "__main__":
    x = np.zeros((5, 3, 15, 15))
    y = generate_batch_equiv(x)
    print(y.shape)
    
    