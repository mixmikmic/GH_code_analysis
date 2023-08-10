import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn, random

from gridWorldEnvironment import GridWorld

# creating gridworld environment
gw = GridWorld(gamma = .9, theta = .5)

def state_action_value(env):
    q = dict()
    for state, action, next_state, reward in env.transitions:
        q[(state, action)] = np.random.normal()
    for action in env.actions:
        q[0, action] = 0
    return q

state_action_value(gw)

def generate_greedy_policy(env, Q):
    pi = dict()
    for state in env.states:
        actions = []
        q_values = []
        prob = []
        
        for a in env.actions:
            actions.append(a)
            q_values.append(Q[state,a])   
        for i in range(len(q_values)):
            if i == np.argmax(q_values):
                prob.append(1)
            else:
                prob.append(0)       
                
        pi[state] = (actions, prob)
    return pi

def e_greedy(env, e, q, state):
    actions = env.actions
    action_values = []
    prob = []
    for action in actions:
        action_values.append(q[(state, action)])
    for i in range(len(action_values)):
        if i == np.argmax(action_values):
            prob.append(1 - e + e/len(action_values))
        else:
            prob.append(e/len(action_values))
    return np.random.choice(actions, p = prob)

def greedy(env, q, state):
    actions = env.actions
    action_values = []
    for action in actions:
        action_values.append(q[state, action])
    return actions[np.argmax(action_values)]

def double_q_learning(env, epsilon, alpha, num_iter):
    Q1, Q2  = state_action_value(env), state_action_value(env)
    for _ in range(num_iter):
        current_state = np.random.choice(env.states)
        while current_state != 0:
            Q = dict()
            for key in Q1.keys():
                Q[key] = Q1[key] + Q2[key]
            current_action = e_greedy(env, epsilon, Q, current_state)
            next_state, reward = env.state_transition(current_state, current_action)
            
            # choose Q1 or Q2 with equal probabilities (0.5)
            chosen_Q = np.random.choice(["Q1", "Q2"])
            if chosen_Q == "Q1":    # when Q1 is chosen
                best_action = greedy(env, Q1, next_state)
                Q1[current_state, current_action] += alpha *                     (reward + env.gamma * Q2[next_state, best_action] - Q1[current_state, current_action])
            else:                    # when Q2 is chosen
                best_action = greedy(env, Q2, next_state)
                Q2[current_state, current_action] += alpha *                     (reward + env.gamma * Q1[next_state, best_action] - Q2[current_state, current_action])
                    
            current_state = next_state
    return Q1, Q2

Q1, Q2 = double_q_learning(gw, 0.2, 0.5, 5000)

# sum Q1 & Q2 elementwise to obtain final Q-values
Q = dict()
for key in Q1.keys():
    Q[key] = Q1[key] + Q2[key]

pi_hat = generate_greedy_policy(gw, Q)

def show_policy(pi, env):
    temp = np.zeros(len(env.states) + 2)
    for s in env.states:
        a = pi_hat[s][0][np.argmax(pi_hat[s][1])]
        if a == "U":
            temp[s] = 0.25
        elif a == "D":
            temp[s] = 0.5
        elif a == "R":
            temp[s] = 0.75
        else:
            temp[s] = 1.0
            
    temp = temp.reshape(4,4)
    ax = seaborn.heatmap(temp, cmap = "prism", linecolor="#282828", cbar = False, linewidths = 0.1)
    plt.show()

### RED = TERMINAL (0)
### GREEN = LEFT
### BLUE = UP
### PURPLE = RIGHT
### ORANGE = DOWN

show_policy(pi_hat, gw)

