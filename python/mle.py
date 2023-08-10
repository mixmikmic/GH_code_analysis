import numpy as np
from scipy.stats import bernoulli 

np.random.seed(123456789)

p_true = 1/2 # this is the value we will try to estimate from the observed data
fp = bernoulli(p_true)

def sample(n=10):
    """
    simulate coin flipping
    """
    return fp.rvs(n) # flip it n times

xs = sample(100) # generate some samples

import sympy
from sympy.abc import x

p = sympy.symbols('p', positive=True)
phi = p ** x * (1 - p) ** (1 - x)
L = np.prod([phi.subs(x, i) for i in xs]) # objective function to maximize
log_L = sympy.expand_log(sympy.log(L))
sol = sympy.solve(sympy.diff(log_L, p), p)[0]

import matplotlib.pyplot as plt

x_space = np.linspace(1/100, 1, 100, endpoint=False)

plt.plot(x_space,
         list(map(sympy.lambdify(p, log_L, 'numpy'), x_space)),
         sol,
         log_L.subs(p, sol),
         'o',
         p_true,
         log_L.subs(p, p_true),
         's',
        )
plt.xlabel('$p$', fontsize=18)
plt.ylabel('Likelihood', fontsize=18)
plt.title('Estimate not equal to true value', fontsize=18)
plt.grid(True)
plt.show()

def estimator_gen(niter=10, ns=100):
    """
    generate data to estimate distribution of maximum likelihood estimator'
    """
    x = sympy.symbols('x', real=True)
    phi = p**x*(1-p)**(1-x)
    for i in range(niter):
        xs = sample(ns) # generate some samples from the experiment
        L = np.prod([phi.subs(x,i) for i in xs]) # objective function to maximize
        log_L = sympy.expand_log(sympy.log(L)) 
        sol = sympy.solve(sympy.diff(log_L, p), p)[0]
        yield float(sol.evalf())
    
entries = list(estimator_gen(100)) # this may take awhile, depending on how much data you want to generate
plt.hist(entries) # histogram of maximum likelihood estimator
plt.title('$\mu={:3.3f},\sigma={:3.3f}$'.format(np.mean(entries), np.std(entries)), fontsize=18)
plt.show()

def estimator_dynamics(ns_space, num_tries = 20):
    for ns in ns_space:
        estimations = list(estimator_gen(num_tries, ns))
        yield np.mean(estimations), np.std(estimations)
    
ns_space = list(range(10, 100, 5))
entries = list(estimator_dynamics(ns_space))
entries_mean = list(map(lambda e: e[0], entries))
entries_std = list(map(lambda e: e[1], entries))

plt.errorbar(ns_space, entries_mean, entries_std, fmt='-o')
plt.show()



