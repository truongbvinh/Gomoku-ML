"""
Author: Vinh Truong
"""

import numpy as np


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

def generate_four_equiv(train, target):
    """
    Takes in a 2D numpy array (training info) and
    the move made (target info) and returns the board
    and move rotated to generate equivalent data

    Returns a tuple of lists;
        First list contains 4 copies of the training info, 
        each rotated 90 more than the preceding element

        Second list contains 4 copies of target info, each
        matching the rotations of the board
    """
    dim_row, dim_col = len(train), len(train[0])
    transform = []
    for x in range(dim_row):
        transform.append(target[x*dim_col:(x+1)*dim_col])
    train_result = [(train)]
    target_result = [(target)]
    for i in range(1, 4):
        train_result.append(np.rot90(train, i))
        target_result.append(np.rot90(transform, i).flatten())
    
    return (train_result, target_result)

def generate_batch_equiv(train_list, target_list):
    train_result = []
    target_result = []

    for train, target in zip(train_list, target_list):
        new_train, new_target = generate_four_equiv(train, target)
        train_result.extend(new_train)
        target_result.extend(new_target)
    
    return (train_result, target_result)

if __name__ == "__main__":
    x = np.array([[1,2,],[3,4]])
    x = [x]
    y = np.array([0.1,0.2,0.3,0.4])
    y = [y]
    train, target = (generate_batch_equiv(x, y))
    for i, j in zip(train, target):
        print(i, j)
        print(type(i), type(j))
    
    