import random
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt

# This is an array, so we can show the improvement
# as increasing number of histories are used
num_hists = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

# Monte Carlo simulation function. This is defined as
# a function so the numba library can be used to speed
# up execution. Otherwise, this would run much slower.
@jit
def MCHist(n_hist):
    score = 0
    for n in range(1, n_hist):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        # Check if the point falls inside the target
        if (x**2 + y**2) <= 1:
            # If so, give it a score of 4
            score += 4
    return score

# Run the simulation, iterating over each number of 
# histories in the num_hists array
results = {}
for n in num_hists:
    results[n] = MCHist(n) / n
    
# Show the results in a table
df = pd.DataFrame.from_dict(results, orient="index")
df

