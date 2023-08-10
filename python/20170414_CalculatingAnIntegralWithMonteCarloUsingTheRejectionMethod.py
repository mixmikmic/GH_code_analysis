import random
from numba import jit
import numpy as np

# Monte Carlo simulation function. This is defined as
# a function so the numba library can be used to speed
# up execution. Otherwise, this would run much slower.
@jit
def MCHist(n_hist, a, b, fmax):
    score = (b - a)*fmax
    tot_score = 0
    for n in range(1, n_hist):
        x = random.uniform(a, b)
        f = random.uniform(0, fmax)
        f_x = np.exp(x**2)
        # Check if the point falls inside the integral      
        if f < f_x:
            tot_score += score
    return tot_score

# Run the simulation
num_hist = 1e8
results = MCHist(num_hist, 0.0, 2.0, 54.6) 
integral_val = round(results / num_hist, 6)
print("The calculated integral is {}".format(integral_val))

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

x = np.linspace(1,4,1000)
y = (np.exp(x)/x) + np.exp(1/x)
plt.plot(x,y)
plt.ylabel('F(x)')
plt.xlabel('x');

@jit
def MCHist2(n_hist, a, b, fmax):
    score = (b - a)*fmax
    tot_score = 0
    for n in range(1, n_hist):
        x = random.uniform(a, b)
        f = random.uniform(0, fmax)
        f_x = (np.exp(x)/x) + np.exp(1/x)
        # Check if the point falls inside the integral      
        if f < f_x:
            tot_score += score
    return tot_score

# Run the simulation
num_hist2 = 1e8
results2 = MCHist2(num_hist2, 1.0, 4.0, 100) 
integral_val2 = round(results2 / num_hist2, 6)
print("The calculated integral is {}".format(integral_val2))

num_hist3 = 1e8
results3 = MCHist2(num_hist2, 1.0, 4.0, 15) 
integral_val3 = round(results3 / num_hist3, 6)
print("The calculated integral is {}".format(integral_val3))

