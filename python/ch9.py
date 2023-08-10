import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm
from numpy.random import choice

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# First choose a bid b (we chose 0.6)
b = 0.6
# then simulate a large number of hypothetical mystery prizes and store them in V
num_trials = 10**5
V = uniform.rvs(size=num_trials)

# Get the average profit conditional on an accepted bid
print(np.mean(V[b > (2/3.)*V]) - b)   # this should be negative regardless of b

print(''.join(np.random.choice(['H', 'T'], size=100, replace=True)))

R = []
num_trials = 10**3
for i in range(num_trials):
    R.append(''.join(np.random.choice(['H', 'T'], size=100, replace=True)))

# Locate 'HH' in strings
T = [s.find('HH') for s in R]
print(np.mean(T) + len('HH'))   # ending position

# Locate 'HT' in strings
T = [s.find('HT') for s in R]
print(np.mean(T) + len('HT'))   # ending position

X = norm.rvs(size=100)   # realizations of r.v. X ~ N(0, 1)
Y = 3 + 5*X + norm.rvs(size=100)   # realizations of r.v. Y = a + bX + eps, where eps ~ N(0, 1)

b = np.cov([X, Y], rowvar=True)[0, 1] / np.var(X)
print(b)
a = np.mean(Y) - b*np.mean(X)
print(a)

plt.figure(figsize=(8, 8))
_ = plt.scatter(X, Y, marker='o', color='black')
_ = plt.plot(X, a + b*X)

