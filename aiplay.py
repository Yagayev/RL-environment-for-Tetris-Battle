
import pygame
from tetris import Tetris
import random
import json
import numpy as np
import sys

spartan = True

if spartan:
    alpha_init = 0.1
    alpha_min = 0.001
    alpha_decay = True
    alpha_decay_factor = 0.999999

    epsilon_init = 0.8
    epsilon_decay = True
    epsilon_min = 0.0001
    epsilon_decay_factor = 0.99999
else:
    alpha_init = 0.01
    alpha_min = 0.001
    alpha_decay = False
    alpha_decay_factor = 0.999999999

    epsilon_init = 0.8
    epsilon_decay = True
    epsilon_min = 0.0001
    epsilon_decay_factor = 0.999999


alpha = alpha_init
epsilon = epsilon_init




# returns scalar, [a,b]X[c,d] = a*c+b*d
def vec_scalaric_mul(vec1, vec2):
    # print("multiply_vec", vec1, vec2)
    som = 0
    for i in range(0,len(vec2)):
        som += vec1[i]*vec2[i]
    return som


def mul_vec(vec1, vec2):
    ans = []
    for i in range(0, len(vec2)):
        ans.append(vec1[i] * vec2[i])
    return ans


def add_vec(vec1, vec2):
    ans = []
    for i in range(0, len(vec2)):
        ans.append(vec1[i] + vec2[i])
    return ans


def sub_vec(vec1, vec2):
    ans = []
    for i in range(0, len(vec2)):
        ans.append(vec1[i] - vec2[i])
    return ans


def abs_vec(vec):
    ans = []
    for i in range(0, len(vec)):
        ans.append(abs(vec[i]))
    return ans


def mul_vec_by_a_acalar(multiplier, state):
    res = []
    for i in range(0, len(state)):
        res.append(multiplier * state[i])
    return res



def normalizeOUT(vec):
        prblematic = False
        for i in range(0,len(vec)):
            if  (-0.0001 < vec[i] < 0.0001):
                prblematic = True
        else:
            normalize_to_one(vec)
        return vec

def normalize_to_one(vec):
    diviser = 0
    for v in vec:
        diviser += abs(v)
    return mul_vec_by_a_acalar(1/diviser, vec)


def create_vectors(size,initVal):
    allVecs = [0]* 53
    i = 0
    for i in range(53):
        singleVec = [0]* size
        for j in range(0 ,size):
            singleVec[j] = initVal
        allVecs[i] = singleVec
        #print(allVecs[i])
    filename = 'vecs.json'
    try:
        with open(filename) as data_file:
            allVecs = json.load(data_file)
    except:
        print("No file inisiall 0")
    finally:
        print("successfully loaded file")

    return allVecs

class EvalActions:

    def __init__(self,size, initVal):
        self.weights = create_vectors(size, initVal)

    def eval_grade(self, state, action):
        if (0 <= action < 53 ):
            return vec_scalaric_mul(self.weights[action], state)
        raise(NonLegalAction("not a valid action "))

    def save_vecs(self):
        with open('vecs.json', 'w') as outfile:
            json.dump(self.weights, outfile, indent=4)



    def normalize3(self, relevent):
        for i in range(0, len(relevent)):
            if (-0.0001 < relevent[i] < 0.0001):
                diviser = 0
                for vec in self.weights:
                    for v in self.vec:
                        diviser += abs(v)

                for vec in self.weights:
                    mul_vec_by_a_acalar(2 / diviser, vec)

    def update(self, state, action, actual_grade ,nextval):
        # based on lecture 6 page page 14
        # print("state predict update", self.weights_left, state)
        expected_grade = self.eval_grade(state, action)
        grade_error = (actual_grade + nextval*0.001 -expected_grade)
        # if(-0.01 < grade_error <0.01 ):
        #     return
        if (0 <= action < 53 ):
            # grad_times_error_vec = mul_vec_by_a_acalar(grade_error, gradient)
            state_times_w = mul_vec(self.weights[action], state)
            error_times_w_states_vec = mul_vec_by_a_acalar(grade_error, state_times_w)
            delta_w = mul_vec_by_a_acalar(alpha, error_times_w_states_vec)
            # print("update StateEval\n",
            #       "\nweights ", self.weights,
            #       "\nstate ", state,
            #       "\nactual_grade ", actual_grade,
            #       "\nexpected_grade ", expected_grade,
            #       "\ngrade_error ", grade_error,
            #       "\nerror_times_w_states_vec ", error_times_w_states_vec,
            #       "\ndelta_w ", delta_w,
            #       "\nadd_vec ", add_vec(self.weights, delta_w))
            self.weights_right = add_vec(self.weights[action], delta_w)
            #self.normalize3(self.weights_right)
        else:
            raise (NonLegalAction("not valid action"))


class NonLegalAction(ValueError):
    pass


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
    for i in range(53):
        evaled = action_evaluator.eval_grade(given_state, i)
        if(evaled > best_action_grade ):
            best_action_grade = evaled
            best_action = i
    return best_action


def observation_to_score(observation):
    return - abs(observation[2])


def act_random():
    return random.randint(0, 52)





# input: tetris state, expecting an array with 2 elements, the first is the board grid
# and the second is an array with the binary 42 numbers representing the 7 shapes(current,
# 5 next and shift), and 3 numbers at the end representing board statistics
# return: a flat array of features
# for each cell i, cell j, and shape k, there are 8 fetures representing the truth table
# of i, j, k
def evaluate_features(state):
    flattened_board = np.array(state[0]).flatten()
    # board_grid = feature_arr_to_grid(flattened_board)
    board_cubes_grid = two_feature_arrs_to_grid(flattened_board, state[1][0:7])
    return board_cubes_grid

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


def get_score(oldCleared, newCleared, isComplete):
    score = 0
    if not isComplete:
        score = 1
    score = score + (newCleared-oldCleared)*10
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
    for i_episode in range(5000):
        observation = env.reset()
        grade = 0
        state = evaluate_features(observation)
        old_state = False
        for t in range(2000000):
            if alpha_decay:
                alpha = max(alpha * alpha_decay_factor, alpha_min)
            if rand:
                env.render()
            prevClear = env.totalCleared
            action = act_epsilon_greedy(state, action_evaluator)

            observation, _, done, _, _ = env.step(action)

            if not done:
                # grade = env.totalCleared -  prevClear
                grade = get_score(prevClear, env.totalCleared, done)
                grad_sum += grade
                nextstate = evaluate_features(observation)
                if old_state:
                    action = act_greedy(nextstate, action_evaluator)
                    nextval = action_evaluator.eval_grade(state, action)
                    action_evaluator.update(old_state, action, grade,nextval)
                old_state = state
                state = nextstate
            # print("reward:", reward)
            if done:
                print("action is", action)
                print("grade_sum is", grad_sum/(iteration+1))
                sys.stdout.flush()
                action_evaluator.update(old_state, action, -500, 0)
                iteration += 1
                grad_sum += grade
                env = train_env
                if iteration % 500 == 0:
                    print(grad_sum/iteration)
                    sys.stdout.flush()
                    iteration = 0
                    grad_sum = 0
                    # env = run_env
                break
    env.close()


if __name__ == '__main__':
    main()
