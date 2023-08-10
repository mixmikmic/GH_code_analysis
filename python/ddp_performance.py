from __future__ import print_function
import numpy as np
from quantecon.markov import random_discrete_dp

def compare_performance(num_states, num_actions, beta, k,
                        suppress_vi=False, random_state=0):
    labels = ['n x m x n', 'n*m x n (dense)', 'n*m x n (sparse)']
    flags = [(False, False), (False, True), (True, True)]  # (sparse, sa_pair)
    ddps = {}
    for label, flag in zip(labels, flags):
        ddps[label] =             random_discrete_dp(num_states, num_actions, beta, k=k,
                               sparse=flag[0], sa_pair=flag[1],
                               random_state=random_state)

    if suppress_vi:
        methods = ['pi', 'mpi']
    else:
        methods = ['vi', 'pi', 'mpi']
    results = {}
    max_iter = 1000
    for ddp in ddps.values():
        ddp.max_iter = max_iter
    k_mpi = 20

    for label in labels:
        results[label] = {method: ddps[label].solve(method=method, k=k_mpi)
                          for method in methods}

    print('(num_states, num_actions) = ({0}, {1})'
          .format(num_states, num_actions))
    print('Number of possible next states for each (s, a) =', k)
    print('beta =', beta)
    print('=====')
    print('Whether the results by pi agree:',
          all([np.array_equal(results[labels[i]]['pi'].sigma,
                              results[labels[2]]['pi'].sigma)
               for i in [0, 1]]))
    print('Whether the answer is correct ({0}, {1}, {2}):'.format(*labels))
    for method in methods:
        if method != 'pi':
            print(method.ljust(3) + ':',
                  [np.array_equal(results[label][method].sigma,
                                  results[label]['pi'].sigma)
                   for label in labels])
    print('Number of iterations ({0}, {1}, {2}):'.format(*labels))
    for method in methods:
        print(method.ljust(3) + ':',
              [results[label][method].num_iter for label in labels])
    print('=====')

    print('Speed comparison ({0}, {1}, {2}):'.format(*labels))
    for method in methods:
        print('***', method, '***')
        for label in labels:
            global ddps, label, method
            get_ipython().magic('timeit ddps[label].solve(method=method)')

seed = 1234  # Set random seed

compare_performance(num_states=100, num_actions=20, beta=0.95, k=3,
                    random_state=seed)

compare_performance(num_states=500, num_actions=20, beta=0.95, k=3,
                    random_state=seed)

compare_performance(num_states=1000, num_actions=20, beta=0.95, k=3,
                    random_state=seed)

compare_performance(num_states=1000, num_actions=50, beta=0.95, k=3,
                    random_state=seed)

compare_performance(num_states=500, num_actions=20, beta=0.95, k=100,
                    random_state=seed)

compare_performance(num_states=500, num_actions=20, beta=0.95, k=50,
                    random_state=seed)

compare_performance(num_states=500, num_actions=20, beta=0.95, k=500,
                    random_state=seed)

compare_performance(num_states=1000, num_actions=100, beta=0.95, k=1,
                    random_state=seed)

compare_performance(num_states=1000, num_actions=200, beta=0.95, k=1,
                    suppress_vi=True, random_state=seed)

import platform
print(platform.platform())

import sys
print(sys.version)

print(np.__version__)

import scipy
print(scipy.__version__)

import numba
print(numba.__version__)



