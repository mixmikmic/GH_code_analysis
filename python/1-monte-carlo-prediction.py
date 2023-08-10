import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn

from gridWorldEnvironment import GridWorld

# creating gridworld environment
gw = GridWorld(gamma = .9, theta = .5)

def generate_random_episode(env):
    episode = []
    done = False
    current_state = np.random.choice(env.states)
    episode.append((current_state, -1))
    while not done:
        action = np.random.choice(env.actions)
        next_state, reward = gw.state_transition(current_state, action)
        episode.append((next_state, reward))
        if next_state == 0:
            done = True
        current_state = next_state
    return episode

generate_random_episode(gw)

def value_array(env):
    return np.zeros(len(env.states)+2)

def first_visit_mc(env, num_iter):
    values = value_array(env)
    returns = dict()
    for state in env.states:
        returns[state] = list()
    
    for i in range(num_iter):
        episode = generate_random_episode(env)
        already_visited = set({0})   # also exclude terminal state (0)
        for s, r in episode:
            if s not in already_visited:
                already_visited.add(s)
                idx = episode.index((s, r))
                G = 0
                j = 1
                while j + idx < len(episode):
                    G = env.gamma * (G + episode[j + idx][1])
                    j += 1
                returns[s].append(G)
                values[s] = np.mean(returns[s])
    return values, returns            

get_ipython().run_cell_magic('time', '', 'values, returns = first_visit_mc(gw, 10000)')

# obtained values
values

def show_values(values):
    values = values.reshape(4,4)
    ax = seaborn.heatmap(values, cmap = "Blues_r", annot = True, linecolor="#282828", linewidths = 0.1)
    plt.show()

show_values(values)

def every_visit_mc(env, num_iter):
    values = value_array(env)
    returns = dict()
    for state in env.states:
        returns[state] = list()
    
    for i in range(num_iter):
        episode = generate_random_episode(env)
        for s, r in episode:
            if s != 0:    # exclude terminal state (0)
                idx = episode.index((s, r))
                G = 0
                j = 1
                while j + idx < len(episode):
                    G = env.gamma * (G + episode[j + idx][1])
                    j += 1
                returns[s].append(G)
                values[s] = np.mean(returns[s])
    return values, returns

get_ipython().run_cell_magic('time', '', 'values, returns = every_visit_mc(gw, 10000)')

# obtained values
values

show_values(values)

