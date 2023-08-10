import numpy as np
from collections import defaultdict

"""
GridWorld 
global variables
"""
n = 5
grid = np.zeros((n, n))

A = [(0, 1), (4, 1), 10]
B = [(0, 3), (2, 3), 5]

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def policy_evaluation(policy, gamma, error, grid):
    for c in range(1000):
        grid_new = np.zeros((n, n))
        # (i, j) covers all possible states (aka s')
        # note that in this problem, the result of taking an action is deterministic, 
        # so there is not need to compute an expectation over all the possible outcomes of taking an action
        # as in (3.12)
        for i in range(n):
            for j in range(n):
                st = (i, j)
                pi = policy[st] # get the prob distrib for state st
                # in the case of the special points A and B, the agent chooses an action randomly
                # but the result is always moving to A' and B' regardless of the action
                # so there is no need to compute a weighted sum over all possible actions as in 
                # the other states that are not A or B
                if st == A[0]:
                    st =  A[1]
                    r = A[2]
                    grid_new[i, j] = 1 * (r + gamma * grid[st[0], st[1]])
                elif st == B[0]:
                    st = B[1]
                    r = B[2]
                    grid_new[i, j] = 1 * (r + gamma * grid[st[0], st[1]])
                else:
                    # sum of rewards(*) over all possible actions from current state (i, j)
                    # (*) weighted by the probability of taking each action under policy pi
                    for aIndex, a in enumerate(actions):
                        r = 0
                        stplus1 = (st[0] + a[0], st[1] + a[1])  # element-wise addition of tuples
                        if stplus1[0]<0 or stplus1[0]>n-1 or stplus1[1]<0 or stplus1[1]>n-1:
                            stplus1 = st
                            r = -1
                        grid_new[i, j] += pi[aIndex] * (r + gamma * grid[stplus1[0], stplus1[1]])

        if np.max(np.abs(grid - grid_new)) < error: break
        grid = grid_new

    return grid


pi = [.25, .25, .25, .25]
policy = {(i,j): pi for j in range(n) for i in range(n)}
gamma=0.9 # discount rate   
error=10^-3
state_values = np.zeros((n, n))
state_values = policy_evaluation(policy, gamma, error, state_values)
print np.round(state_values, decimals=2)

def policy_improvement(state_values, gamma, policy):
    policy_stable = True
    # (4.3):
    # (i, j) covers all possible states (state s at time t)
    for i in range(n):
        for j in range(n):
            st = (i, j)
            # in the case of the special points A and B, the agent chooses an action as per policy
            # but the result is always moving to A' and B' deterministically regardless of the action
            # so the optimal policy used by the agent to choose the action doesn't matter in A and B
            if st == A[0]:
                st = A[1]
                r = A[2]
            elif st == B[0]:
                st = B[1]
                r = B[2]
            else:
                # maximize for all actions 
                action_values = []
                for a in actions:
                    # find state t+1 (given s and a)
                    stplus1 = (st[0] + a[0], st[1] + a[1])  # element-wise addition of tuples
                    
                    # find the reward for transitioning to state t+1
                    r = 0
                    if stplus1[0]<0 or stplus1[0]>n-1 or stplus1[1]<0 or stplus1[1]>n-1:
                        stplus1 = st
                        r = -1
                    # compute the action value with the recursive formula (the algorithm is not recursive thanks to DP)
                    action_values.append(1 * (r + gamma * state_values[stplus1[0], stplus1[1]]))
                    
                # update the policy (associate prob distribution to current state)
                pi = [.0 for _ in actions]
                pi[np.argmax(action_values)] = 1.
                
                if pi != policy[st]:
                    policy_stable = False
                    policy[st] = pi
                    
    return policy, policy_stable

def policy_iteration(gamma, error):
    state_values = np.zeros((n, n))
    pi = [.25, .25, .25, .25]
    policy = {(i,j): pi for j in range(n) for i in range(n)}
    
    for i in range(1000):
        state_values = policy_evaluation(policy, gamma, error, state_values)
        
        policy, policy_stable =         policy_improvement(state_values, gamma, policy)
        
        print
        print i
        print np.round(state_values, decimals=2)
        
        if policy_stable:
            break
            
    return state_values, policy

gamma=0.9 # discount rate  
error=10^-3
state_values, policy = policy_iteration(gamma, error)

print "Note that for points A and B the action doesn't matter because all actions lead to A' and B'."
print "In those points the policy is the original uniform distribution, in this case argmax pick the first: actions[0]"
for i in range(n):
    for j in range(n):
        print "(%d,%d): " % (i, j), actions[np.argmax(policy[(i, j)])]



