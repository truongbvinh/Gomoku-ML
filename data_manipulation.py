"""
Author: Vinh Truong
"""

import numpy as np

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
    
    """
    [0, 1, 2, 3
     4, 5, 6, 7
     8, 9, 10,11
     12,13,14,15]
    """
    dim_row, dim_col = len(train), len(train[0])
    transform = [target[x*dim_col:(x+1)*dim_col] for x in range(dim_row)]
    train_result = [].append(train)
    target_result = [].append(target)
    for i in range(1, 4):
        train_result.append(np.rot90(train, i))
        target_result.append(np.rot90(transform, i).flatten)
    
    return (train_result, target_result)

def generate_batch_equiv(train_list, target_list):
    train_result = []
    target_result = []

    for train, target in zip(train_list, target_list):
        new_train, new_target = generate_four_equiv(train, target)
        train_result.extend(new_train)
        target_result.extend(new_target)
    
    return (train_result, target_result)