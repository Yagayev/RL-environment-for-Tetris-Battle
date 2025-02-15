
import pygame
from tetris import Tetris
import random
import json
import numpy as np
import sys
import csv
import warnings

np.seterr(all='warn')
alpha_init = 0.00001
alpha_min = 0.00000001
alpha_decay = True
alpha_decay_factor = 0.99999999999

epsilon_init = 0.8
epsilon_decay = True
epsilon_min = 0.01
epsilon_decay_factor = 0.9999995


alpha = alpha_init
epsilon = epsilon_init
gamma = 0.9

height_punish = 0.3
height_reward = 5

def create_vectors(size, initVal):
    allVecs = [0] * 52
    for i in range(52):
        singleVec = [0] * size
        for j in range(0, size):
            singleVec[j] = initVal
        allVecs[i] = singleVec
        #print(allVecs[i])
    filename = 'vecs.json'
    try:
        with open(filename) as data_file:
            allVecs = json.load(data_file)
            print("successfully loaded file")
    except:
        print("No file; initial 1")
    finally:
        print("done trying to load")
    allVecs = [np.array(x, dtype=np.float64) for x in allVecs]
    return allVecs

class EvalActions:

    def __init__(self,size, initVal):
        self.weights = create_vectors(size, initVal)

    def eval_grade(self, state, action):

        state = np.array(state)
        # norm = np.linalg.norm(state)
        # state = state/norm
        # print("state", state, "norm", norm, "")
        if (0 <= action < 52 ):
            return np.dot(self.weights[action], state)
        raise(NonLegalAction("not a valid action "))

    def save_vecs(self):
        lst = [x.tolist() for x in self.weights]
        with open('vecs.json', 'w') as outfile:
            json.dump(lst, outfile, indent=4)

    def update(self, state, action, actual_grade, nextval):
        global gamma
        # based on lecture 6 page page 14
        # print("state predict update", self.weights_left, state)
        expected_grade = self.eval_grade(state, action)
        grade_error = (actual_grade + nextval*gamma- expected_grade)
        # if(-0.01 < grade_error <0.01 ):
        #     return
        if (0 <= action < 52 ):
            # grad_times_error_vec = mul_vec_by_a_acalar(grade_error, gradient)
            try:
                state_times_w = np.multiply(self.weights[action], state)
                error_times_w_states_vec = grade_error * state_times_w
                delta_w = alpha * error_times_w_states_vec
            except Warning:
                print("weights:", self.weights[action], "\nstate:", state)
            # print("update StateEval\n",
            #       "\nweights ", self.weights,
            #       "\nstate ", state,
            #       "\nactual_grade ", actual_grade,
            #       "\nexpected_grade ", expected_grade,
            #       "\ngrade_error ", grade_error,
            #       "\nerror_times_w_states_vec ", error_times_w_states_vec,
            #       "\ndelta_w ", delta_w,
            #       "\nadd_vec ", add_vec(self.weights, delta_w))
            self.weights[action] = self.weights[action] + delta_w
            norm = np.linalg.norm(self.weights)
            self.weights = [weight / norm for weight in self.weights]
            #self.normalize3(self.weights_right)
        else:
            raise (NonLegalAction("not valid action"))


class NonLegalAction(ValueError):
    pass


def act_epsilon_greedy(given_state, action_evaluator, perv_best):
    global epsilon, epsilon_decay_factor
    rand_seed = random.random()

    if rand_seed > epsilon:
        if perv_best != None:
            ans = perv_best
        else:
            ans, _ = act_greedy(given_state, action_evaluator)
    else:
        ans = act_random()
    if epsilon_decay:
        epsilon = max(epsilon * epsilon_decay_factor, epsilon_min)
        epsilon_decay_factor = epsilon_decay_factor*alpha_decay_factor
    return ans


def act_greedy(given_state, action_evaluator):
    best_action = 0
    best_action_grade = 0
    for i in range(52):
        evaled = action_evaluator.eval_grade(given_state, i)
        if(evaled > best_action_grade ):
            best_action_grade = evaled
            best_action = i
    return best_action, best_action_grade


def observation_to_score(observation):
    return - abs(observation[2])


def act_random():
    return random.randint(0, 51)

def observtion_to_squares(lines):
    ans = []
    for i in range(0, len(lines)-1):
        first =lines[i]
        sec = lines[i+1]
        for j in range(0, len(first)-1):
            ans +=[[first[j],first[j+1],sec[j],sec[j+1]]]
    return ans


def evaluate_features(state):
    board = [line.flatten() for line in state[0]]
    line_grid = []
    for line in board:
        line_grid += preceptions_to_feacher_array(line)
    cols = transpose(board)
    col_pairs = []
    for col in cols:
        col_pairs += all_niot(col, 2)
    col_grid = []
    for pair in col_pairs:
        col_grid += preceptions_to_feacher_array(pair)

    squares = observtion_to_squares(board)
    square_grid = []
    for square in squares:
        square_grid += preceptions_to_feacher_array(square)
    flattened_board = np.array(state[0]).flatten()
    board_cubes_grid = np.array(two_feature_arrs_to_grid(flattened_board, state[1][0:7]))
    ans = np.append(line_grid, col_grid)
    ans = np.append(ans, square_grid)
    ans = np.append(ans, board_cubes_grid)
    ans = np.append(ans, state[0].flatten())
    ans = np.append(ans, state[1])
    ans = np.append(ans, [hight(board)])
    ans[-1] = ans[-1] / 20
    ans[-2] = ans[-2] / 200
    ans[-3] = ans[-3] / 200
    ans[-4] = ans[-4] / 200
    return ans



# input: tetris state, expecting an array with 2 elements, the first is the board grid
# and the second is an array with the binary 42 numbers representing the 7 shapes(current,
# 5 next and shift), and 3 numbers at the end representing board statistics
# return: a flat array of features
# for each cell i, cell j, and shape k, there are 8 fetures representing the truth table
# of i, j, k
# def evaluate_features(state):
#     flattened_board = np.array(state[0]).flatten()
#     # board_grid = feature_arr_to_grid(flattened_board)
#     board_cubes_grid = two_feature_arrs_to_grid(flattened_board, state[1][0:7])
#     return board_cubes_grid

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


def all_niot(arr, n):
    return all_niot_impl(arr, [], n)

def make_decimal(vec):
    ans = 0
    revVec = reversed(vec)
    pos = 0
    for member in revVec:
        ans = ans + member* pow(2, pos)
        pos += 1
    return ans


def preceptions_to_feacher_array(collection):
    #collection is a binar veclor.
    #we lock at it as abinar number the corispondig feacher is the same number in unar representsion +1
    grindize = pow(2,len(collection))
    ans = [0] * grindize
    dec = make_decimal(collection)
    ans[dec] = 1
    return ans

def transpose(arr):
    return [[row[i] for row in arr] for i in range(len(arr[0]))]


def all_niot_impl(arr, acc, n):
    if n == 0:
        return [acc]
    if len(arr) == n:
        return [acc+arr]
    if len(arr) + len(acc) < n:
        return []
    first = arr[0]
    rest = arr[1:]
    withcurr = all_niot_impl(rest, acc+[first], n-1)
    without = all_niot_impl(rest, acc, n)
    return withcurr + without

def hight(metrix):
    ans = len(metrix)
    for line in metrix:
        if max(line) == 0:
            ans -=1
        else:
            return ans
    return ans

def get_score(oldCleared, newCleared,oldstate,newstate ,isComplete):
    score = 0
    if not isComplete:
        score = 1
    score = score + (newCleared-oldCleared)*10
    statelen = len(newstate)
    height_delta = (newstate[statelen-1] -oldstate[statelen-1])  # positive is bad
    # score -=(newstate[statelen-3] -oldstate[statelen-3])*bumbingPunish
    holes_delta =(newstate[statelen-2] -oldstate[statelen-2])  # positive is bad
    if height_delta > 0 and holes_delta > 0:
        score -= height_punish
    if height_delta <= 0 and holes_delta <= 0:
        score += height_reward
    return score

def main():
    global alpha, epsilon
    # run_env = Tetris(action_type='grouped', is_display=True)
    train_env = Tetris(action_type='grouped', is_display=False)
    env = train_env
    observation = env.reset()
    state = evaluate_features(observation)
    # print(state)

    action_evaluator = EvalActions(len(state), 1.0)
    iteration = 0
    grad_sum = 0
    rand = False
    total_score = 0
    prev_best = None
    for i_episode in range(50000000):
        observation = env.reset()
        grade = 0
        state = evaluate_features(observation)
        old_state = []
        for t in range(2000000):
            if alpha_decay:
                alpha = max(alpha * alpha_decay_factor, alpha_min)
            if rand:
                env.render()
            prevClear = env.totalCleared
            action = act_epsilon_greedy(state, action_evaluator, prev_best)

            observation, _, done, _, _ = env.step(action)

            if not done:
                # grade = env.totalCleared -  prevClear
                nextstate = evaluate_features(observation)


                if any(old_state):
                    grade = get_score(prevClear, env.totalCleared, old_state, nextstate, done)
                    grad_sum += grade
                    prev_best, nextval = act_greedy(nextstate, action_evaluator)
                    # nextval = action_evaluator.eval_grade(state, action)
                    action_evaluator.update(old_state, prev_best, grade,nextval)
                old_state = state
                state = nextstate
            # print("reward:", reward)
            if done:
                action_evaluator.update(old_state, action, -500, 0)
                iteration += 1
                grad_sum += grade
                env = train_env
                if iteration % 50 == 0:
                    print("game number", i_episode, "avg:", grad_sum/iteration, "epsilon:", epsilon, "alpha:", alpha)
                    sys.stdout.flush()
                    f = open('resolts.csv', 'a', newline='')
                    with f:
                        writer = csv.writer(f)
                        writer.writerows([[grad_sum/iteration]])
                    action_evaluator.save_vecs()
                    # sys.stdout.flush()
                    iteration = 0
                    grad_sum = 0
                    # env = run_env
                break
    env.close()
    action_evaluator.save_vecs()


if __name__ == '__main__':
    main()
