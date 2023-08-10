import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Since the Markov assumption requires that the future 
# state only depends on the current state, we will keep
# track of the current state during each iteration.
# "0" is low, "1" is normal, and "2" is high
def MCDegradeSim(t_prob, d_per_state, d_thresh):
    # As the number of states increase, the initial state
    # below might make a difference, so be careful.
    cur_state = 1
    cur_deg = 0
    res = []
    deg = []
    # "while True" is usually a bad idea, but I know that the while
    # loop must terminate, because the accrued degradation is 
    # always positive
    while True:
        rn = random.random()
        # Contrary to the previous blog post, this will be
        # done with much more coding efficiency.
        if rn<=t_prob[cur_state][0]:
            cur_state = 0
            cur_deg += d_per_state[0]
        # Remember that it's the cumulative probability
        elif rn<=(t_prob[cur_state][0] + t_prob[cur_state][1]):
            cur_state = 1
            cur_deg += d_per_state[1]
        else:
            cur_state = 2
            cur_deg += d_per_state[2]
        # Save the results to an array
        res.append(cur_state)
        deg.append(cur_deg)
        # If the degradation is above the threshold, the 
        # simulation is done
        if cur_deg>d_thresh:
            break
    return res
# Transition probability matrix, taken from the image above
tpm = [[0.8, 0.19, 0.01],[0.01, 0.98, 0.01],[0.01, 0.2, 0.79]]
# Don't cheat and look at this! This is the degradation 
# accrued per state and the damage threshold
dps = [0.5, 0.1, 1.5]
deg_thresh = 100
# Run and plot the results
res = MCDegradeSim(tpm, dps, deg_thresh)
plt.plot(res)
plt.title('Transition Probability=' + str(tpm))
plt.xlabel('Iteration')
plt.ylabel('State');

# The new transition probability matrix
tpm2 = [[0.9, 0.09, 0.01],[0.05, 0.90, 0.05],[0.01, 0.1, 0.89]]
res2 = MCDegradeSim(tpm2, dps, deg_thresh)
plt.plot(res2)
plt.title('Transition Probability=' + str(tpm2))
plt.xlabel('Iteration')
plt.ylabel('State');

num_failures = 100
# Run the MCDegradeSim function for num_failures
res_array = []
for ii in range(num_failures):
    res = MCDegradeSim(tpm, dps, deg_thresh)
    res_array.append(res)

import numpy as np
# Keep track of the transitions from/to each state 
trans_matrix = np.zeros((3,3))
# "hist" is the time history for a single equipment
for hist in res_array:
    # Iterate over each state in the time history,
    # and find the transitions between an old state
    # and a new state.
    for ii, state in enumerate(hist):
        old_state = hist[ii-1]
        new_state = hist[ii]
        trans_matrix[old_state, new_state] += 1
# To translate the matrix into probabilities, 
# divide by the total
trans_prob = trans_matrix/(sum(trans_matrix))
print(trans_prob.transpose())

from sklearn.linear_model import LinearRegression as LR
# "X"- Keep track of the number of times a state occurs
# over an entire time history
x = np.zeros((num_failures,3))
for ii, hist in enumerate(res_array):
    # Bincount will sum the number of times that a
    # state occurs in the history
    x[ii, :] = np.bincount(hist)
# "Y" is always 100% at failure
y = 100*np.ones((100,3))
# Now, perform linear regression on the above data
lr_model = LR(fit_intercept=False)
lr_model.fit(x, y)
print(lr_model.coef_[0])

