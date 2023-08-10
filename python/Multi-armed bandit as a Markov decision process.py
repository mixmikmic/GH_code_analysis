# Standard plotting setup
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
plt.style.use('ggplot')

import itertools
import numpy as np
from pprint import pprint

def sorted_values(dict_):
    return [dict_[x] for x in sorted(dict_)]

def solve_bmab_value_iteration(N_arms, M_trials, gamma=1,
                               max_iter=10, conv_crit = .01):
    util = {}
    
    # Initialize every state to utility 0.
    state_ranges = [range(M_trials+1) for x in range(N_arms*2)]
    # The reward state
    state_ranges.append(range(2))
    for state in itertools.product(*state_ranges):
        # Some states are impossible to reach.
        if sum(state[:-1]) > M_trials:
            # A state with the total of alphas and betas greater than 
            # the number of trials.
            continue
            
        if sum(state[:-1:2]) == 0 and state[-1] == 1:
            # A state with a reward but alphas all equal to 0.
            continue
            
        if sum(state[:-1:2]) == M_trials and state[-1] == 0:
            # A state with no reward but alphas adding up to M_trials.
            continue
            
        if sum(state[:-1]) == 1 and sum(state[:-1:2]) == 1 and state[-1] == 0:
            # A state with an initial reward according to alphas but not according
            # the reward index
            continue
            
        util[state] = 0
    
    # Main loop.
    converged = False
    new_util = util.copy()
    opt_actions = {}
    for j in range(max_iter):
        # Line 5 of value iteration
        for state in util.keys():
            reward = state[-1]
            
            # Terminal state.
            if sum(state[:-1]) == M_trials:
                new_util[state] = reward
                continue
            
            values = np.zeros(N_arms)
            
            # Consider every action
            for i in range(N_arms):
                # Successes and failure for this state.
                alpha = state[i*2]
                beta  = state[i*2+1]
                
                # Two possible outcomes: either that arm gets rewarded,
                # or not.
                # Transition to unrewarded state:
                state0 = list(state)
                state0[-1] = 0
                state0[2*i+1] += 1
                state0 = tuple(state0)
                
                # The probability that we'll transition to this unrewarded state.
                p_state0 = (beta + 1) / float(alpha + beta + 2)
                
                # Rewarded state.
                state1 = list(state)
                state1[-1] = 1
                state1[2*i] += 1
                state1 = tuple(state1)
                
                p_state1 = 1 - p_state0
                try:
                    value = gamma*(util[state0]*p_state0 + 
                                   util[state1]*p_state1)
                except KeyError,e:
                    print state
                    print state0
                    print state1
                    raise e
                    
                #print state0, util[state0], p_state0
                #print state1, util[state1], p_state1
                values[i] = value
                
            #print state, values, reward
            new_util[state] = reward + np.max(values)
            opt_actions[state] = np.argmax(values)
            
        # Consider the difference between the new util
        # and the old util.
        max_diff = np.max(abs(np.array(sorted_values(util)) - np.array(sorted_values(new_util))))
        util = new_util.copy()
        
        print "Iteration %d, max diff = %.5f" % (j, max_diff)
        if max_diff < conv_crit:
            converged = True
            break
            
        #pprint(util)
            
    if converged:
        print "Converged after %d iterations" % j
    else:
        print "Not converged after %d iterations" % max_iter
        
    return util, opt_actions

util, opt_actions = solve_bmab_value_iteration(2, 2, max_iter=5)

opt_actions

util

2*.5*2.0/3.0 + .5/3.0 + .5*.5

util, opt_actions = solve_bmab_value_iteration(2, 3, max_iter=5)
opt_actions

util, opt_actions = solve_bmab_value_iteration(2, 4, max_iter=6)

M_trials = 16
get_ipython().magic('time util, opt_actions = solve_bmab_value_iteration(2, M_trials, max_iter=M_trials+2)')

# Create a design matrix related to the optimal strategies.
X = []
y = []
seen_keys = {}
for key, val in opt_actions.iteritems():
    if key[:-1] in seen_keys:
        # We've already seen this, continue.
        continue
        
    alpha0 = float(key[0] + 1)
    beta0  = float(key[1] + 1)
    alpha1 = float(key[2] + 1)
    beta1  = float(key[3] + 1)
    
    if alpha0 == alpha1 and beta0 == beta1:
        # We're in a perfectly symmetric situtation, skip this then.
        continue
        
    seen_keys = key[:-1]
    
    # Standard results for the Beta distribution.
    # https://en.wikipedia.org/wiki/Beta_distribution
    mean0 = alpha0/(alpha0 + beta0)
    mean1 = alpha1/(alpha1 + beta1)
    
    std0  = np.sqrt(alpha0*beta0 / (alpha0 + beta0 + 1)) / (alpha0 + beta0)
    std1  = np.sqrt(alpha1*beta1 / (alpha1 + beta1 + 1)) / (alpha1 + beta1)
    
    t = alpha0 + beta0 + alpha1 + beta1
    X.append([mean0,mean1,std0,std1,t,1,alpha0 - 1,beta0 - 1,alpha1 - 1,beta1 - 1])
    y.append(val)
    
X = np.array(X)
y = np.array(y)

from sklearn.linear_model import LogisticRegression

the_model = LogisticRegression(C=100.0)
X_ = X[:,:2]
the_model.fit(X_,y)
y_pred = the_model.predict(X_)

print ("Greedy: %.4f%% of moves are incorrect" % ((np.mean(abs(y_pred-y)))*100))
print the_model.coef_

the_model = LogisticRegression(C=100.0)
X_ = X[:,:4]
the_model.fit(X_,y)
y_pred = the_model.predict(X_)

print ("UCB: %.4f%% of moves are incorrect" % ((np.mean(abs(y_pred-y)))*100))
print the_model.coef_

the_model = LogisticRegression(C=100000.0)
X_ = X[:,:4]
X_ = np.hstack((X_,(X[:,4]).reshape((-1,1))*X[:,2:4]))
the_model.fit(X_,y)
y_pred = the_model.predict(X_)

print ("UCB X time: %.4f%% of moves are incorrect" % ((np.mean(abs(y_pred-y)))*100))
print the_model.coef_

