import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn, random

from gridWorldEnvironment import GridWorld

# creating gridworld environment
gw = GridWorld(gamma = .9)

def state_action_value(env):
    q = dict()
    for state, action, next_state, reward in env.transitions:
        q[(state, action)] = np.random.normal()
    return q

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
    return actions, prob

def generate_e_greedy_policy(env, e, Q):
    pi = dict()
    for state in env.states:
        pi[state] = e_greedy(env, e, Q, state)
    return pi

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

# function for tree backup algorithm
def avg_over_actions(pi, Q, state):
    actions, probs = pi[state]
    q_values = np.zeros(4)
    for s, a in Q.keys():
        if s == state:
            q_values[actions.index(a)] = Q[s,a]
    return np.dot(q_values, probs)

def n_step_tree_backup(env, epsilon, alpha, n, num_iter, learn_pi = True):
    Q = state_action_value(env)
    Q_, pi_, delta = dict(), dict(), dict()  
    pi = generate_e_greedy_policy(env, epsilon, Q) 

    for _ in range(num_iter):
        current_state = np.random.choice(env.states)
        action = np.random.choice(b[current_state][0], p = b[current_state][1])
        state_trace, action_trace, reward_trace  = [current_state], [action], [0]
        Q_[0] = Q[current_state, action]
        t, T = 0, 10000
        while True:
            if t < T:    
                next_state, reward = env.state_transition(current_state, action)
                state_trace.append(next_state)
                reward_trace.append(reward)
                if next_state == 0:
                    T = t + 1
                    delta[t] = reward - Q_[t]
                else:  
                    delta[t] = reward + env.gamma * avg_over_actions(pi, Q, next_state) - Q_[t]
                    action = np.random.choice(pi[next_state][0], p = pi[next_state][1])
                    action_trace.append(action)
                    Q_[t+1] = Q[next_state, action]
                    pi_[t+1] = pi[next_state][1][pi[next_state][0].index(action)]
                    
            tau = t - n + 1
            if tau >= 0:
                Z = 1
                G = Q_[tau]
                for i in range(tau, min([tau + n -1, T-1])):
                    G += Z * delta[i]
                    Z *= env.gamma * pi_[i+1]
                Q[state_trace[tau], action_trace[tau]] += alpha * (G - Q[state_trace[tau], action_trace[tau]])
                if learn_pi:
                    pi[state_trace[tau]] = e_greedy(env, epsilon, Q, state_trace[tau])
            current_state = next_state    
#             print(state_trace, action_trace, reward_trace)
            
            if tau == (T-1):
                break
            t += 1
            
    return pi, Q

pi, Q = n_step_tree_backup(gw, 0.2, 0.5, 1, 10000)

### RED = TERMINAL (0)
### GREEN = LEFT
### BLUE = UP
### PURPLE = RIGHT
### ORANGE = DOWN

show_policy(pi, gw)

