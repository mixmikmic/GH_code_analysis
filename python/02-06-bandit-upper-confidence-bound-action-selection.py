from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

n = 2000
k = 10
models = np.random.normal(0.0, 1.0, size=(n, k))

models[0:1,:]

# the bandit returns the actual reward from the true model
def bandit(row, a):
    r = models[row, a] + np.random.normal()
    return r

from math import log, sqrt
# upper-confidence-bound method
def UCB(Q, N, t, c):
    a = np.argmax(Q + c * np.log(sqrt(t+1) / N))
    return a

def update_rule(old_estimate, target, step_size):
    new_estimate = old_estimate + step_size * (target - old_estimate)
    return new_estimate

c = 0.5 # confidence level, smaller values mean higher confidence (larger values allow more exploration)
Qs = np.ones(shape=(n, k))
num_steps = 100
 
# ratio for plotting performance
ratio_est_vs_opt = np.zeros(shape=(n, num_steps))
# accumulators for plotting performance
rewards_accum =   np.zeros(shape=(n, num_steps))
opt_rewards_accum =   np.zeros(shape=(n, num_steps)) + 1/10**6 # avoid division by zero at step zero
    
# for each model
for i in range(n):
    # action counters
    N = np.ones(k) * 1/10**6 # avoid division by zero  in UCB computation
    
    # 1 run
    for t in range(num_steps):       
        # select action, based on estimated action-values, with upper-confidence-bound method
        a = UCB(Qs[i,:], N, t, c)
        
        # act and collect the actual reward from the bandit
        reward = bandit(i, a)

        # update our estimate of the action value 
        N[a] += 1
        Qs[i, a] = update_rule(Qs[i, a], reward, 1/N[a])
              
        # store the accumulators to calculate the ratio of epsilon-greedy vs optimal at each step for plotting
        if t > 0:
            rewards_accum[i, t] = rewards_accum[i, t-1] + reward
            opt_rewards_accum[i, t] = opt_rewards_accum[i, t-1] + bandit(i, np.argmax(models[i,:]))

# Compute ratio of cumulative rewards
# The stationary bandit test bed often contains commulative rewards that are close to zero
# I average over the 2000 models before computing the ratio

# mean along rows (avg of each step over all models)
avg_rewards_accum = np.mean(rewards_accum, 0)
avg_opt_rewards_accum = np.mean(opt_rewards_accum, 0)

#  average performance over all models
avg_ratio_est_vs_opt = avg_rewards_accum / avg_opt_rewards_accum

plt.plot(avg_ratio_est_vs_opt)

c = 2 # confidence level, smaller values mean higher confidence (larger values allow more exploration)
Qs = np.ones(shape=(n, k))
num_steps = 100
 
# ratio for plotting performance
ratio_est_vs_opt = np.zeros(shape=(n, num_steps))
# accumulators for plotting performance
rewards_accum =   np.zeros(shape=(n, num_steps))
opt_rewards_accum =   np.zeros(shape=(n, num_steps)) + 1/10**6 # avoid division by zero at step zero
    
# for each model
for i in range(n):
    # action counters
    N = np.ones(k) * 1/10**6 # avoid division by zero  in UCB computation
    
    # 1 run
    for t in range(num_steps):       
        # select action, based on estimated action-values, with upper-confidence-bound method
        a = UCB(Qs[i,:], N, t, c)
        
        # act and collect the actual reward from the bandit
        reward = bandit(i, a)

        # update our estimate of the action value 
        N[a] += 1
        Qs[i, a] = update_rule(Qs[i, a], reward, 1/N[a])
              
        # store the accumulators to calculate the ratio of epsilon-greedy vs optimal at each step for plotting
        if t > 0:
            rewards_accum[i, t] = rewards_accum[i, t-1] + reward
            opt_rewards_accum[i, t] = opt_rewards_accum[i, t-1] + bandit(i, np.argmax(models[i,:]))

# Compute ratio of cumulative rewards
# The stationary bandit test bed often contains commulative rewards that are close to zero
# I average over the 2000 models before computing the ratio

# mean along rows (avg of each step over all models)
avg_rewards_accum = np.mean(rewards_accum, 0)
avg_opt_rewards_accum = np.mean(opt_rewards_accum, 0)

#  average performance over all models
avg_ratio_est_vs_opt = avg_rewards_accum / avg_opt_rewards_accum

plt.plot(avg_ratio_est_vs_opt)



