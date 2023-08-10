def print_policy(pi, P):
    arrs = {k:v for k,v in enumerate(('<', 'v', '>', '^'))}
    for key, value in pi.items():
        print("| ", end="")
        if P[key][0][0][0] == 1.0:
            print("    ", end=" ")
        else:
            print(str(key).zfill(2), arrs[value], end=" ")
        if (key + 1) % np.sqrt(len(pi)) == 0: print("|")

P = {0: {0: [(0.3333333333333333, 0, 0.0, False),
   (0.3333333333333333, 0, 0.0, False),
   (0.3333333333333333, 4, 0.0, False)],
  1: [(0.3333333333333333, 0, 0.0, False),
   (0.3333333333333333, 4, 0.0, False),
   (0.3333333333333333, 1, 0.0, False)],
  2: [(0.3333333333333333, 4, 0.0, False),
   (0.3333333333333333, 1, 0.0, False),
   (0.3333333333333333, 0, 0.0, False)],
  3: [(0.3333333333333333, 1, 0.0, False),
   (0.3333333333333333, 0, 0.0, False),
   (0.3333333333333333, 0, 0.0, False)]},
 1: {0: [(0.3333333333333333, 1, 0.0, False),
   (0.3333333333333333, 0, 0.0, False),
   (0.3333333333333333, 5, 0.0, True)],
  1: [(0.3333333333333333, 0, 0.0, False),
   (0.3333333333333333, 5, 0.0, True),
   (0.3333333333333333, 2, 0.0, False)],
  2: [(0.3333333333333333, 5, 0.0, True),
   (0.3333333333333333, 2, 0.0, False),
   (0.3333333333333333, 1, 0.0, False)],
  3: [(0.3333333333333333, 2, 0.0, False),
   (0.3333333333333333, 1, 0.0, False),
   (0.3333333333333333, 0, 0.0, False)]},
 2: {0: [(0.3333333333333333, 2, 0.0, False),
   (0.3333333333333333, 1, 0.0, False),
   (0.3333333333333333, 6, 0.0, False)],
  1: [(0.3333333333333333, 1, 0.0, False),
   (0.3333333333333333, 6, 0.0, False),
   (0.3333333333333333, 3, 0.0, False)],
  2: [(0.3333333333333333, 6, 0.0, False),
   (0.3333333333333333, 3, 0.0, False),
   (0.3333333333333333, 2, 0.0, False)],
  3: [(0.3333333333333333, 3, 0.0, False),
   (0.3333333333333333, 2, 0.0, False),
   (0.3333333333333333, 1, 0.0, False)]},
 3: {0: [(0.3333333333333333, 3, 0.0, False),
   (0.3333333333333333, 2, 0.0, False),
   (0.3333333333333333, 7, 0.0, True)],
  1: [(0.3333333333333333, 2, 0.0, False),
   (0.3333333333333333, 7, 0.0, True),
   (0.3333333333333333, 3, 0.0, False)],
  2: [(0.3333333333333333, 7, 0.0, True),
   (0.3333333333333333, 3, 0.0, False),
   (0.3333333333333333, 3, 0.0, False)],
  3: [(0.3333333333333333, 3, 0.0, False),
   (0.3333333333333333, 3, 0.0, False),
   (0.3333333333333333, 2, 0.0, False)]},
 4: {0: [(0.3333333333333333, 0, 0.0, False),
   (0.3333333333333333, 4, 0.0, False),
   (0.3333333333333333, 8, 0.0, False)],
  1: [(0.3333333333333333, 4, 0.0, False),
   (0.3333333333333333, 8, 0.0, False),
   (0.3333333333333333, 5, 0.0, True)],
  2: [(0.3333333333333333, 8, 0.0, False),
   (0.3333333333333333, 5, 0.0, True),
   (0.3333333333333333, 0, 0.0, False)],
  3: [(0.3333333333333333, 5, 0.0, True),
   (0.3333333333333333, 0, 0.0, False),
   (0.3333333333333333, 4, 0.0, False)]},
 5: {0: [(1.0, 5, 0, True)],
  1: [(1.0, 5, 0, True)],
  2: [(1.0, 5, 0, True)],
  3: [(1.0, 5, 0, True)]},
 6: {0: [(0.3333333333333333, 2, 0.0, False),
   (0.3333333333333333, 5, 0.0, True),
   (0.3333333333333333, 10, 0.0, False)],
  1: [(0.3333333333333333, 5, 0.0, True),
   (0.3333333333333333, 10, 0.0, False),
   (0.3333333333333333, 7, 0.0, True)],
  2: [(0.3333333333333333, 10, 0.0, False),
   (0.3333333333333333, 7, 0.0, True),
   (0.3333333333333333, 2, 0.0, False)],
  3: [(0.3333333333333333, 7, 0.0, True),
   (0.3333333333333333, 2, 0.0, False),
   (0.3333333333333333, 5, 0.0, True)]},
 7: {0: [(1.0, 7, 0, True)],
  1: [(1.0, 7, 0, True)],
  2: [(1.0, 7, 0, True)],
  3: [(1.0, 7, 0, True)]},
 8: {0: [(0.3333333333333333, 4, 0.0, False),
   (0.3333333333333333, 8, 0.0, False),
   (0.3333333333333333, 12, 0.0, True)],
  1: [(0.3333333333333333, 8, 0.0, False),
   (0.3333333333333333, 12, 0.0, True),
   (0.3333333333333333, 9, 0.0, False)],
  2: [(0.3333333333333333, 12, 0.0, True),
   (0.3333333333333333, 9, 0.0, False),
   (0.3333333333333333, 4, 0.0, False)],
  3: [(0.3333333333333333, 9, 0.0, False),
   (0.3333333333333333, 4, 0.0, False),
   (0.3333333333333333, 8, 0.0, False)]},
 9: {0: [(0.3333333333333333, 5, 0.0, True),
   (0.3333333333333333, 8, 0.0, False),
   (0.3333333333333333, 13, 0.0, False)],
  1: [(0.3333333333333333, 8, 0.0, False),
   (0.3333333333333333, 13, 0.0, False),
   (0.3333333333333333, 10, 0.0, False)],
  2: [(0.3333333333333333, 13, 0.0, False),
   (0.3333333333333333, 10, 0.0, False),
   (0.3333333333333333, 5, 0.0, True)],
  3: [(0.3333333333333333, 10, 0.0, False),
   (0.3333333333333333, 5, 0.0, True),
   (0.3333333333333333, 8, 0.0, False)]},
 10: {0: [(0.3333333333333333, 6, 0.0, False),
   (0.3333333333333333, 9, 0.0, False),
   (0.3333333333333333, 14, 0.0, False)],
  1: [(0.3333333333333333, 9, 0.0, False),
   (0.3333333333333333, 14, 0.0, False),
   (0.3333333333333333, 11, 0.0, True)],
  2: [(0.3333333333333333, 14, 0.0, False),
   (0.3333333333333333, 11, 0.0, True),
   (0.3333333333333333, 6, 0.0, False)],
  3: [(0.3333333333333333, 11, 0.0, True),
   (0.3333333333333333, 6, 0.0, False),
   (0.3333333333333333, 9, 0.0, False)]},
 11: {0: [(1.0, 11, 0, True)],
  1: [(1.0, 11, 0, True)],
  2: [(1.0, 11, 0, True)],
  3: [(1.0, 11, 0, True)]},
 12: {0: [(1.0, 12, 0, True)],
  1: [(1.0, 12, 0, True)],
  2: [(1.0, 12, 0, True)],
  3: [(1.0, 12, 0, True)]},
 13: {0: [(0.3333333333333333, 9, 0.0, False),
   (0.3333333333333333, 12, 0.0, True),
   (0.3333333333333333, 13, 0.0, False)],
  1: [(0.3333333333333333, 12, 0.0, True),
   (0.3333333333333333, 13, 0.0, False),
   (0.3333333333333333, 14, 0.0, False)],
  2: [(0.3333333333333333, 13, 0.0, False),
   (0.3333333333333333, 14, 0.0, False),
   (0.3333333333333333, 9, 0.0, False)],
  3: [(0.3333333333333333, 14, 0.0, False),
   (0.3333333333333333, 9, 0.0, False),
   (0.3333333333333333, 12, 0.0, True)]},
 14: {0: [(0.3333333333333333, 10, 0.0, False),
   (0.3333333333333333, 13, 0.0, False),
   (0.3333333333333333, 14, 0.0, False)],
  1: [(0.3333333333333333, 13, 0.0, False),
   (0.3333333333333333, 14, 0.0, False),
   (0.3333333333333333, 15, 1.0, True)],
  2: [(0.3333333333333333, 14, 0.0, False),
   (0.3333333333333333, 15, 1.0, True),
   (0.3333333333333333, 10, 0.0, False)],
  3: [(0.3333333333333333, 15, 1.0, True),
   (0.3333333333333333, 10, 0.0, False),
   (0.3333333333333333, 13, 0.0, False)]},
 15: {0: [(1.0, 15, 0, True)],
  1: [(1.0, 15, 0, True)],
  2: [(1.0, 15, 0, True)],
  3: [(1.0, 15, 0, True)]}}

# import gym
# P = gym.make('FrozenLake-v0').env.P

import numpy as np

# we first define the policy we will evaluate
LEFT, DOWN, RIGHT, UP = range(4)
pi = {
    0:RIGHT, 1:LEFT, 2:DOWN, 3:UP,
    4:LEFT, 5:LEFT, 6:RIGHT, 7:LEFT,
    8:UP, 9:DOWN, 10:UP, 11:LEFT,
    12:LEFT, 13:RIGHT, 14:DOWN, 15:LEFT
}

# Now we define the policy evaluation method
def policy_evaluation(pi, P, gamma=0.9, theta=1e-10):
    V = np.zeros(len(pi))
    
    while True:
        max_delta = 0
        old_V = V.copy()

        for s in range(len(P)):
            V[s] = 0
            for prob, new_state, reward, done in P[s][pi[s]]:
                if done:
                    value = reward
                else:
                    value = reward + gamma * old_V[new_state]
                V[s] += prob * value
            max_delta = max(max_delta, abs(old_V[s] - V[s]))
        if max_delta < theta:
            break
    return V.copy()

# Run it and display the results
V = policy_evaluation(pi, P)
V

# define the algorithm
def policy_improvement(pi, V, P, gamma=0.9):

    for s in range(len(V)):
        Qs = np.zeros(len(P[0]), dtype=np.float64)
        for a in range(len(P[s])):
            for prob, new_state, reward, done in P[s][a]:
                if done:
                    value = reward
                else:
                    value = reward + gamma * V[new_state]
                Qs[a] += prob * value
        pi[s] = np.argmax(Qs)
    return pi.copy()

# run and return the results
new_pi = policy_improvement(pi, V, P)
new_pi

print_policy(new_pi, P)

# lets define the algorithm
def policy_iteration(P, gamma=0.9):

    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = {s:a for s, a in enumerate(random_actions)}

    while True:
        old_pi = pi.copy()

        V = policy_evaluation(pi, P, gamma)
        pi = policy_improvement(pi, V, P, gamma)

        if old_pi == pi:
            break

    return V, pi

# we call it passing an MDP, and return the 
# optimal policy and state-value function
V_best, pi_best = policy_iteration(P)
V_best, pi_best

print_policy(pi_best, P)

# define the value iteration algorithm
def value_iteration(P, gamma=0.9, theta = 1e-10):

    V = np.random.random(len(P))
    while True:
        max_delta = 0
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

        for s in range(len(P)):
            v = V[s]
            for a in range(len(P[s])):
                for prob, new_state, reward, done in P[s][a]:
                    if done:
                        value = reward
                    else:
                        value = reward + gamma * V[new_state]
                    Q[s][a] += prob * value
            V[s] = np.max(Q[s])
            max_delta = max(max_delta, abs(v - V[s]))
        if max_delta < theta:
            break
    pi = {s:a for s, a in enumerate(np.argmax(Q, axis=1))}
    return V, pi

# run it and return the optimal policy and state-value function
V_best, pi_best = value_iteration(P, gamma=0.9)
V_best, pi_best

print_policy(pi_best, P)

# change reward function
reward_goal, reward_holes, reward_others = 1, -1, -0.01
goal, hole = 15, [5, 7, 11, 12]
for s in range(len(P)):
    for a in range(len(P[s])):
        for t in range(len(P[s][a])):
            values = list(P[s][a][t])
            if values[1] == goal:
                values[2] = 1
                values[3] = False
            elif values[1] in hole:
                values[2] = -1
                values[3] = False
            else:
                values[2] = -0.01
                values[3] = False
            if s in hole or s == goal:
                values[2] = 0
                values[3] = True
            P[s][a][t] = tuple(values)

# change transition function
prob_action, drift_right, drift_left = 0.8, 0.1, 0.1
for s in range(len(P)):
    for a in range(len(P[s])):
        for t in range(len(P[s][a])):
            if P[s][a][t][0] == 1.0:
                continue
            values = list(P[s][a][t])
            if t == 0:
                values[0] = drift_left
            elif t == 1:
                values[0] = prob_action
            elif t == 2:
                values[0] = drift_right
            P[s][a][t] = tuple(values)

