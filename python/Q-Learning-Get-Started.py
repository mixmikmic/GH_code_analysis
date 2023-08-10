import numpy as np
import random

# initial
q = np.zeros([6, 6])
q = np.matrix(q)

r = np.array([[-1, -1, -1, -1, 0, -1], [-1, -1, -1, 0, -1, 100], [-1, -1, -1, 0, -1, -1], [-1, 0, 0, -1, 0, -1], [0, -1, -1, 0, -1, 100], [-1, 0, -1, -1, 0, 100]])
r = np.matrix(r)

gamma = 0.8

# training
for i in range(100):
    # one episode
    state = random.randint(0, 5)
    while (state != 5):
        # choose positive r-value action randomly
        r_pos_action = []
        for action in range(6):
            if r[state, action] >= 0:
                r_pos_action.append(action)
        
        next_state = r_pos_action[random.randint(0, len(r_pos_action) - 1)]
        q[state, next_state] = r[state, next_state] + gamma * q[next_state].max()
        state = next_state

# verify
for i in range(10):
    # one episode
    print("episode: " + str(i + 1))
    
    # random initial state
    state = random.randint(0, 5)
    print("the robot borns in " + str(state) + ".")
    count = 0
    while (state != 5):
        # prevent endless loop
        if count > 20:
            print('fails')
            break
            
        # choose maximal q-value action randomly
        q_max = -100
        for action in range(6):
            if q[state, action] > q_max:
                q_max = q[state, action]
            
        q_max_action = []
        for action in range(6):
            if q[state, action] == q_max:
                q_max_action.append(action)
                
        next_state = q_max_action[random.randint(0, len(q_max_action) - 1)]
        
        print("the robot goes to " + str(next_state) + '.')
        state = next_state
        count = count + 1



