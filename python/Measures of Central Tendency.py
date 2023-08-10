# Two useful statistical libraries
import scipy.stats as stats
import numpy as np

# We'll use these two data sets as examples
x1 = [1, 2, 2, 3, 4, 5, 5, 7]
x2 = x1 + [100]

print 'Mean of x1:', sum(x1), '/', len(x1), '=', np.mean(x1)
print 'Mean of x2:', sum(x2), '/', len(x2), '=', np.mean(x2)

print 'Median of x1:', np.median(x1)
print 'Median of x2:', np.median(x2)

# Scipy has a built-in mode function, but it will return exactly one value
# even if two values occur the same number of times, or if no value appears more than once
print 'One mode of x1:', stats.mode(x1)[0][0]

# So we will write our own
def mode(l):
    # Count the number of times each element appears in the list
    counts = {}
    for e in l:
        try:
            counts[e] += 1
        except(KeyError):
            counts[e] = 1
    # Return the elements that appear the most times
    maxcount = 0
    modes = {}
    for (key, value) in counts.iteritems():
        if value > maxcount:
            maxcount = value
            modes = {key}
        elif value == maxcount:
            modes.add(key)
    if maxcount > 1 or len(l) == 1:
        return list(modes)
    return 'No mode'
    
print 'All of the modes of x1:', mode(x1)

# Get return data for an asset and compute the mode of the data set
start = '2014-01-01'
end = '2015-01-01'
pricing = get_pricing('SPY', fields='price', start_date=start, end_date=end)
returns = pricing.pct_change()[1:]
print 'Mode of returns:', mode(returns)

# Since all of the returns are distinct, we use a frequency distribution to get an alternative mode.
# np.histogram returns the frequency distribution over the bins as well as the endpoints of the bins
hist, bins = np.histogram(returns, 20) # Break data up into 20 bins
maxfreq = max(hist)
# Find all of the bins that are hit with frequency maxfreq, then print the intervals corresponding to them
print 'Mode of bins:', [(bins[i], bins[i+1]) for i, j in enumerate(hist) if j == maxfreq]

# Use scipy's gmean function to compute the geometric mean
print 'Geometric mean of x1:', stats.gmean(x1)
print 'Geometric mean of x2:', stats.gmean(x2)

# Add 1 to every value in the returns array and then compute R_G
ratios = returns + np.ones(len(returns))
R_G = stats.gmean(ratios) - 1
print 'Geometric mean of returns:', R_G

T = len(returns)
init_price = pricing[0]
final_price = pricing[T]
print 'Initial price:', init_price
print 'Final price:', final_price
print 'Final price as computed with R_G:', init_price*(1 + R_G)**T

print 'Harmonic mean of x1:', stats.hmean(x1)
print 'Harmonic mean of x2:', stats.hmean(x2)

