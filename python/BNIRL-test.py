get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import learning

matplotlib.style.use('mitch-exp')

data = np.array([(0, y) for y in range(20)] + 
                [(x, 20) for x in range(20)] +
                [(20, y) for y in range(20)[::-1]] +
                [(x, 0) for x in range(20)[::-1]])
data

noisy_data = data + np.random.randn(*data.shape)/4

plt.scatter(noisy_data[:, 0], noisy_data[:, 1])

actions = np.diff(noisy_data, axis=0)
actions.shape

noisy_data.shape

fig = plt.figure()
args = np.split(noisy_data, 2, axis=1) + np.split(actions, 2, axis=1)
plt.title('State-Action Observations')

plt.scatter(*np.split(noisy_data, 2, axis=1), c='r')
plt.quiver(*args, color='b')

from jupyter_extras import log_progress

noisy_data_trim = noisy_data[:-1]
noisy_data_trim.shape

# Initialize the subgoal partition labels all to zero, and size to track convergence
partitions = [0]*(len(noisy_data_trim))
iters = 100
partition_dist = np.zeros((iters, len(noisy_data_trim)))
size = np.zeros((iters))

# Run BNIRL for _ iterations # eta works around 0.0000001
for i in log_progress(range(iters)):
    partitions = learning.bnirl_sampling_3(noisy_data_trim, partitions, actions, verbose=False, eta=0.0000001)
    partition_dist[i] = np.array(partitions)
    size[i] = (len(set(partitions)))

# Not too useful...
plt.figure()
plt.plot(size)
plt.title('Number of Clusters at Each Iteration')

from scipy import stats

modes = np.zeros(partition_dist.shape)
# modes = np.zeros(len(states_condense))
brn = 0

# Store mode for each state over Gibbs sample sweeps up to ith iteration
for j in range(len(partition_dist)): # 0-200
    for i in range(partition_dist.shape[1]): # 0-161
        modes[j, i] = stats.mode(partition_dist[:j+1, i])[0][0]

# for i, gibbs_samples in enumerate(partition_dist[brn:].T):
#     modes[i], _ = stats.mode(gibbs_samples)

mode_set = set([int(mode) for mode in modes[-1]])
counts = [(mode, np.count_nonzero(modes[-1]==mode)) for mode in mode_set]
sorted(counts, key=lambda x: x[1])[::-1]

plt.scatter(*np.split(noisy_data_trim[list(mode_set)], 2, axis=1))



