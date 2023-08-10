get_ipython().magic('matplotlib inline')
import numpy as np
from numpy import diag
import numpy.random as rand
from numpy.linalg import norm, inv, eigvals

import sys
from IPython import display
from tqdm import tqdm
from ipyparallel import Client

import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain, repeat

import models
from util import *
from viz import *

plt.rcParams['figure.figsize'] = (12.0, 8.0)

def adjustNorm(A, multiplier):
    '''
    Set the diagonal elements of A to be equal to the sum of their row
    times the multiplier.
    '''
    np.fill_diagonal(A, 0)
    rowSums = np.sum(A, 0)
    A = A + (diag(rowSums) * multiplier)
    return row_stochastic(A)

# Set simulation parameters
rand.seed(49448)
max_rounds = 10000
N = 10
NETWORK_REPS = 40

# Create 14 identical multipliers for each distinct multiplier value
multipliers = list(chain.from_iterable(repeat(mul, NETWORK_REPS) for mul in np.linspace(1, 1e-5, 20)))
# Create Networks
A = np.ones((N, N))
networks = [adjustNorm(A, mul) for mul in multipliers]
print('Generated {0} networks'.format(len(networks)))

# Initialize ipyparallel
variables = dict(max_rounds=max_rounds, Inf=np.Inf)
v, dv = parallel_init('/home/user/opinions-research/', profile='ssh', variables=variables)

def run_model(A):
    import models
    return models.meetFriend_matrix_nomem(A, max_rounds, norm_type=Inf)

result = parallel_map(v, run_model, networks)

# Save for later
np.savetxt('random1.txt', result)

# Open saved data
result = np.loadtxt('random1.txt')

norms = np.array([norm(A-diag(diag(A)), np.Inf) for A in networks])
result = np.array(result)
norms = np.round(norms, decimals=2)
sns.pointplot(norms, result)
plt.xlabel('$||A||_{\infty}$')
plt.ylabel('$||(I-A)^{-1}B - R||_{\infty}$')
plt.title('Distance from Equilibrium as a variable of $||A||_{\infty}$')
plt.show()

rand.seed(4002)

N = 10
A = np.ones((N, N))
max_rounds = 1e6
multipliers = np.linspace(0.1, 1.0, 6)
networks = [adjustNorm(A, m) for m in multipliers]
print 'Infinite Norms = '
for network in networks:
    print(norm(network-diag(diag(network)), np.Inf))

# Initialize ipyparallel
v, dv = parallel_init('/home/user/opinions-research/', profile='ssh', variables=dict(max_rounds=max_rounds, norm_type=np.Inf))

def run_model(A):
    import models
    return models.meetFriend_matrix(A, max_rounds, norm_type=norm_type)
    
result = parallel_map(v, run_model, networks)

np.savetxt('randommatrix2.txt', result)

for idx, op in enumerate(result):
    net_norm = '{:.2f}'.format(norm(networks[idx]-diag(diag(networks[idx])), np.Inf))
    plt.plot(np.log10(op), label='$||A||_{\infty}$ = ' + net_norm)
plt.xlabel('Rounds')
plt.ylabel('Log10(distance)')
plt.title('Log10 of the Distance from the Equilibrium')
plt.legend()
plt.show()

