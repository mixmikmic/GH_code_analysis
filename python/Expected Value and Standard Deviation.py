from __future__ import print_function
from auquanToolbox.dataloader import load_data_nologs
import numpy as np
import scipy.stats as stats

# Let's say the random variables x1 and x2 have the following values
x1 = [10,9,8,5,6,7,4,3,2]
x2 = x1 + [100]

print ('Mean of x1:', sum(x1), '/', len(x1), '=', np.mean(x1))
print ('Mean of x2:', sum(x2), '/', len(x2), '=', np.mean(x2))

print('Median of x1:', np.median(x1))
print('Median of x2:', np.median(x2))

def mode(l):
    # Count the number of times each element appears in the list
    counts = {}
    for e in l:
        if e in counts:
            counts[e] += 1
        else:
            counts[e] = 1
            
    # Return the elements that appear the most times
    maxcount = 0
    modes = {}
    for key in counts:
        if counts[key] > maxcount:
            maxcount = counts[key]
            modes = {key}
        elif counts[key] == maxcount:
            modes.add(key)
            
    if maxcount > 1 or len(l) == 1:
        return list(modes)
    return 'No mode'
    
print ('All of the modes of x1:', mode(x1))

# Use scipy's gmean function to compute the geometric mean
print ('Geometric mean of x1:', stats.gmean(x1))
print ('Geometric mean of x2:', stats.gmean(x2))

print ('Harmonic mean of x1:', stats.hmean(x1))
print ('Harmonic mean of x2:', stats.hmean(x2))

print('Variance of x1:', np.var(x1))
print('Standard deviation of x1:', np.std(x1))
print('Variance of x2:', np.var(x2))
print('Standard deviation of x2:', np.std(x2))

