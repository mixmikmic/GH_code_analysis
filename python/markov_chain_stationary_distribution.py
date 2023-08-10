import numpy as np
import collections
import random

# [bull, bear, stagnant][bull, bear, stagnant]
index_to_name = ['bull', 'bear', 'stagnant']
transition_matrix = np.array([[0.9, 0.075, 0.025],[0.15, 0.8, 0.05], [0.25, 0.25, 0.5]])

# bull -> bear
transition_matrix[0][1]

# Find stationary distribution
def get_next_state(state):
    r = random.random()
    prob_sum = 0
    new_state = len(transition_matrix) - 1
    for i in range(len(transition_matrix)):
        prob_sum += transition_matrix[state][i]
        if prob_sum >= r:
            new_state = i
            break
    return new_state
        

num_samples = 1000000
state = 0
count = collections.Counter()
for i in range(num_samples):
    count[state] += 1
    state = get_next_state(state)

for i in range(len(index_to_name)):
    print('{} stationary probability: {}'.format(index_to_name[i], count[i] / num_samples))



