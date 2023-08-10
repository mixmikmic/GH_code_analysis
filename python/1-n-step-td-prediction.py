import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn

from gridWorldEnvironment import GridWorld

# creating gridworld environment
gw = GridWorld(gamma = .9, theta = .5)

def initialize_values(env):
    V = dict()
    for state in env.states:
        V[state] = np.random.normal()
    return V

def generate_random_policy(env):
    pi = dict()
    for state in env.states:
        actions = []
        prob = []
        for action in env.actions:
            actions.append(action)
            prob.append(0.25)
        pi[state] = (actions, prob)
    return pi

def n_step_td_prediction(env, alpha = .5, n = 3, num_iter = 100):
    V = initialize_values(env)
    pi = generate_random_policy(env)
    
    for _ in range(num_iter):
        state_trace, action_trace, reward_trace = list(), list(), list()
        current_state = np.random.choice(env.states)
        state_trace.append(current_state)
        t, T = 0, 10000
        while True:
            if t < T:
                action = np.random.choice(pi[current_state][0], p = pi[current_state][1])
                action_trace.append(action)
                next_state, reward = env.state_transition(current_state, action)
                state_trace.append(next_state)
                reward_trace.append(reward)
                if next_state == 0:
                    T = t + 1
                    
            tau = t - n + 1   # tau designates the time step of estimate being update
            if tau >= 0:
                
                G = 0
                for i in range(tau+1, min([tau+n, T]) + 1):
                    G += (env.gamma ** (i - tau - 1)) * reward_trace[i-1]
                if tau + n < T: 
                    G += (env.gamma ** n) * V[state_trace[tau + n]]
                V[state_trace[tau]] += alpha * (G - V[state_trace[tau]])
            
            # terminating condition
            if tau == (T - 1):
                break
            current_state = next_state
            t += 1
    return V

values = n_step_td_prediction(gw, num_iter = 10000)

values

def show_values(V):
    env = GridWorld()
    values = np.zeros(len(env.states) + 2)    
    for k in V.keys():
        values[k] = V[k]
    values = values.reshape(4,4)
    ax = seaborn.heatmap(values, cmap = "Blues_r", annot = True, linecolor="#282828", linewidths = 0.1)
    plt.show()

show_values(values)

