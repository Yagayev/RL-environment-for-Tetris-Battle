import numpy as np
import time
# input: tetris state, expecting an array with 2 elements, the first is the board grid
# and the second is an array with the binary 42 numbers representing the 7 shapes(current,
# 5 next and shift), and 3 numbers at the end representing board statistics
# return: a flat array of features
# for each cell i, cell j, and shape k, there are 8 fetures representing the truth table
# of i, j, k
def evaluate_features(state):
    t0 = time.time()
    flattened_board = np.array(state[0]).flatten()
    board_grid = feature_arr_to_grid(flattened_board)
    # cubes_grid = two_feature_arrs_to_grid(state[1][0:7], state[1][7:14])
    # cubes_grid = two_feature_arrs_to_grid(cubes_grid, state[1][14:21])
    # cubes_grid = two_feature_arrs_to_grid(cubes_grid, state[1][14:21])
    board_cubes_grid = two_feature_arrs_to_grid(flattened_board, state[1][0:7])
    t1 = time.time()
    delta = t1 - t0
    print("evaluate_features ", delta)
    return board_grid+board_cubes_grid

#
def feature_pair_to_2d_grid(f1,f2):
    ans = [0,0,0,0]
    if(f1 > 0):
        if(f2 > 0):
            ans[3] = 1
        else:
            ans[2] = 1
    else:
        if (f2 > 0):
            ans[1] = 1
        else:
            ans[0] = 1
    return ans

def feature_arr_to_grid(arr):
    ans = []
    for i in range(1, len(arr)):
        for j in range(i+1, len(arr)):
            ans = ans + feature_pair_to_2d_grid(arr[i],arr[j])
    return ans

def two_feature_arrs_to_grid(arr1, arr2):
    ans = []
    for i in arr1:
        for j in arr2:
            ans = ans + feature_pair_to_2d_grid(i, j)
    return ans