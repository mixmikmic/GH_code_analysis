import numpy as np
import matplotlib.pyplot as plt
import gym
from scipy import stats as ss
get_ipython().magic('matplotlib inline')

#
# Observation dimensionality - 4: (x, x', theta, theta')
# Action space is discrete
# State space  is continuous
#

env = gym.make('CartPole-v0')
eps = 0.3

NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
NUM_ACTIONS = env.action_space.n  # 0 or 1
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

import math
STATE_BOUNDS[1] = (-0.5, 0.5)
STATE_BOUNDS[3] = (-math.radians(50), math.radians(50))
ACTION_INDEX = len(NUM_BUCKETS)

q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

NUM_EPISODES = 1000
MAX_T = 250
STREAK_TO_END = 120
SOLVED_T = 199
DEBUG_MODE = False

import random

def select_action(state, eps):
    action = None
    if random.random() < eps:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))


def state_to_bucket(state):
    bucket_indice = []
    bucket_index = None
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            offset = scaling * STATE_BOUNDS[i][0]
            bucket_index = int(round(state[i] * scaling - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


def run():    
    discount_factor = 0.99
    num_streaks = 0
    
    for episode in range(NUM_EPISODES):
        exploration_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
        
        obv = env.reset()
        state_0 = state_to_bucket(obv)
        for t in range(MAX_T):
            env.render()
            action = select_action(state_0, exploration_rate)
            obv, reward, done, _ = env.step(action)
            state = state_to_bucket(obv)
            best_q = np.amax(q_table[state])  # get max along the axis _state_
            
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor*best_q                                                             - q_table[state_0 + (action,)])
            state_0 = state
            
            if DEBUG_MODE:
                print("\nEpisode: {}".format(episode))
                print("t: {}".format(t))
                print("Action: {}".format(action))
                print("State: {}".format(state))
                print("Reward: {}".format(reward))
                print("Best Q: {}".format(best_q))
                print("Exploration rate: {}".format(exploration_rate))
                print("Learning rate: {}".format(learning_rate))
                print("Streaks: {}".format(num_streaks))
                print("<<<----|--------|---->>>")
                
            if done:
                print("Episode {} finished after {} timesteps".format(episode, t))
                if t >= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break
                
        if num_streaks > STREAK_TO_END:
            break

run()
env.close()





