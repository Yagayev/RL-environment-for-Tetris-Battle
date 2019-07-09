
import pygame
from tetris import Tetris
import random
import json
import numpy as np
import sys
import csv
import warnings

import math
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.seterr(all='warn')
alpha_init = 0.01
alpha_min = 0.0001
alpha_decay = True
alpha_decay_factor = 0.9999999999

epsilon_init = 1.0
epsilon_decay = True
epsilon_min = 0.01

epsilon_decay_factor = 2 ** -13


epsilon_decay_slowdown_at = 0.35
epsilon_slower_decay = 0.99999995


alpha = alpha_init
epsilon = epsilon_init
gamma = 0.9
state_size = 245

# rewards and punishments
# all punishments are REDUCED from the score
height_and_holes_punish =0.05
height_without_holes_reward = 1
height_reward = 0.5
holes_reward = 0.5
survival_reward = 0.1
death_punish = 10
line_pop_bonus = 5
filename = 'bigger_alpha.json'

no_rotations = list(range(0, 52, 4))
one_rotation = no_rotations + list(range(1, 52, 2))
all_actions = list(range(0,52))

def create_vectors(size, initVal, serial,actionCount):
    allVecs = [0] * actionCount
    for i in range(actionCount):
        singleVec = [0] * size
        for j in range(0, size):
            singleVec[j] = random.uniform(0.1, 0.9)
        allVecs[i] = singleVec
        #print(allVecs[i])
    print("feature vector length = ", size)
    sys.stdout.flush()

    try:
        with open(filename+serial) as data_file:
            allVecs = json.load(data_file)
            print("successfully loaded file")
    except:
        print("No file; initial 1")
    finally:
        print("done trying to load")
    allVecs = [np.array(x, dtype=np.float64) for x in allVecs]
    return allVecs

class EvalActions:

    def __init__(self,size, initVal, serial):
        movmebtCount = relevent_movse(serial)
        self.weights = create_vectors(size, initVal, serial,len(movmebtCount))
        self.serial = serial

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
        with open(filename+self.serial, 'w') as outfile:
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

def relevent_movse(shape):
        if shape == 0 or shape == 4 or shape == 5 : #shape is line (----) or z (****|||___)
            return one_rotation
        if shape == 1: #shape is a squre (ם)
            return no_rotations
        return all_actions #shpe is an L or a plus


def act_epsilon_greedy(given_state, action_evaluator):
    global epsilon
    rand_seed = random.random()

    if rand_seed > epsilon:
        ans = act_greedy(given_state, action_evaluator)
    else:
        ans = act_random()
    if epsilon_decay:
        epsilon = max(epsilon * epsilon_decay_factor, epsilon_min)
    return ans


def act_greedy(given_state, action_evaluator):
    best_action = 0
    best_action_grade = 0
    for i in range(52):
        evaled = action_evaluator.eval_grade(given_state, i)
        if(evaled > best_action_grade ):
            best_action_grade = evaled
            best_action = i
    return best_action


def observation_to_score(observation):
    return - abs(observation[2])


def act_random():
    return random.choice(range(52))

def observtion_to_squares(lines):
    ans = []
    for i in range(0, len(lines)-1):
        first =lines[i]
        sec = lines[i+1]
        for j in range(0, len(first)-1):
            ans +=[[first[j],first[j+1],sec[j],sec[j+1]]]
    return ans


def evaluate_features(state):
    flattened_board = np.array(state[0]).flatten()
    # board_grid = feature_arr_to_grid(flattened_board)
    #board_cubes_grid = flattened_board +  state[1]
    return np.append(flattened_board , state[1])



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



def obs_to_shape_num(obs):
    for i in range(0,7):
        if obs[1][i] == 1:
            return i


def get_score(oldCleared, newCleared,oldstate,newstate ,isComplete):
    score = 0
    score = score + (newCleared-oldCleared)*line_pop_bonus
    statelen = len(newstate)
    height_delta = (newstate[statelen-1] - oldstate[statelen-1])  # positive is bad
    holes_delta = (newstate[statelen-2] - oldstate[statelen-2])  # positive is bad
    if height_delta > 0 and holes_delta > 0:
        score -= height_and_holes_punish
    if height_delta <= 0 and holes_delta <= 0:
        score += height_without_holes_reward
    if height_delta <= 0:
        score += height_reward
    if height_delta <= 0:
        score += holes_reward
    return score



def calc_height_and_holes(state):
    board = [line.flatten() for line in state[0]]
    height = hight(board)
    holes = state[1][-1]
    return height, holes

class scoreCalc():
        def __init__(self, env):
            self.env = env
            self.oldBoom = 0
            self.old_height = 0
            self.old_holes = 0


        def getScore(self, holes, height, done):
            if done:
                # nullify
                self.oldBoom = 0
                self.old_height = 0
                self.old_holes = 0

                return -death_punish

            score = survival_reward + (self.env.totalCleared - self.oldBoom)*line_pop_bonus
            self.oldBoom = self.env.totalCleared

            holes_delta = holes - self.old_holes
            height_delta = height - self.old_height
            if height_delta > 0 and holes_delta > 0:
                score -= height_and_holes_punish
            if height_delta <= 0 and holes_delta <= 0:
                score += height_without_holes_reward
            if height_delta <= 0:
                score += height_reward
            if height_delta <= 0:
                score += holes_reward

            return score

        def zerofy(self):
            self.oldBoom = 0


def preprocess_state(state):
    return np.reshape(state, [1, state_size])


class TetrisAgent:
    def __init__(self, n_episodes=1000000, n_win_ticks=2000, max_env_steps=None, gamma=gamma, epsilon=epsilon_init, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay_factor, alpha=alpha_init, alpha_decay=alpha_decay_factor, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        train_env = Tetris(action_type='grouped', is_display=False)
        self.env = train_env 
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None:
            self.env._max_episode_steps = max_env_steps
        observation = self.env.reset()
        # state = evaluate_features(observation)
        # Init model
        self.model = Sequential()
        self.model.add(Dense(52 , input_dim=state_size, activation='tanh'))#input-observ
        self.model.add(Dense(1024, activation='tanh'))
        self.model.add(Dense(1024, activation='tanh'))
        self.model.add(Dense(52, activation='linear')) #outpot-actions
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        return act_random() if (np.random.random() <= self.epsilon) else np.argmax(self.model.predict(state))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min,self.epsilon-self.epsilon_decay)

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        self.update_epsilon()

    def train(self):
        scores = deque(maxlen=100)
        survive = deque(maxlen=100)
        popped = deque(maxlen=100)
        grader = scoreCalc(self.env)
        for e in range(self.n_episodes):
            grader.zerofy()
            state = evaluate_features(self.env.reset())
            state = preprocess_state(state)
            done = False
            i = 0
            scoreCount = 0
            while not done:
                action = self.choose_action(state)
                next_state, _, done, _, _ = self.env.step(action)
                height, holes = calc_height_and_holes(next_state)
                next_state = evaluate_features(next_state)
                reward = grader.getScore(height=height, holes=holes, done=done)
                sys.stdout.flush()
                scoreCount += reward
                next_state = preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
            popped.append(self.env.totalCleared)
            scores.append(scoreCount)
            survive.append(self.env.step_cnt)
            mean_score = np.mean(scores)
            mean_surv = np.mean(survive)
            mean_popped = np.mean(popped)
            if mean_surv >= self.n_win_ticks and e >= 100:
                if not self.quiet:
                    print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                print("last scores:", mean_score, scores)
                print("last survive:", mean_surv, survive)
                print("+++++++++++++++++++++++++++++++++")
                sys.stdout.flush()
                return e - 100
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks. with epsilon {} '.format(e, mean_score,self.epsilon))
                print("last scores:", mean_score, scores)
                print("last survive:", mean_surv, survive)
                print("popped:", mean_popped, popped)
                print("+++++++++++++++++++++++++++++++++")
                sys.stdout.flush()


            self.replay(self.batch_size)
        
        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

    def play(self):
        self.env = Tetris(action_type='grouped', is_display=True)
        scores = deque(maxlen=100)
        survive = deque(maxlen=100)
        popped = deque(maxlen=100)
        grader = scoreCalc(self.env)
        for e in range(self.n_episodes):
            grader.zerofy()
            state = evaluate_features(self.env.reset())
            state = preprocess_state(state)
            done = False
            i = 0
            scoreCount = 0
            while not done:
                action = self.choose_action(state, 0.01)
                next_state, _, done, _, _ = self.env.step(action)
                height, holes = calc_height_and_holes(next_state)
                next_state = evaluate_features(next_state)
                reward = grader.getScore(height=height, holes=holes, done=done)
                scoreCount += reward
                next_state = preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
            popped.append(self.env.totalCleared)
            scores.append(scoreCount)
            survive.append(self.env.step_cnt)
            mean_score = np.mean(scores)
            mean_surv = np.mean(survive)
            mean_popped = np.mean(popped)
            if mean_surv >= self.n_win_ticks and e >= 100:
                if not self.quiet:
                    print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                print("last scores:", mean_score, scores)
                print("last survive:", mean_surv, survive)
                print("+++++++++++++++++++++++++++++++++")
                sys.stdout.flush()
                return e - 100
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks. with epsilon {} '.format(e, mean_score,self.epsilon))
            print("last scores:", mean_score, scores)
            print("last survive:", mean_surv, survive)
            print("popped:", mean_popped, popped)
            print("+++++++++++++++++++++++++++++++++")
            sys.stdout.flush()

            self.replay(self.batch_size)
        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e


if __name__ == '__main__':
    agent = TetrisAgent()

    agent.train()
    agent.play()