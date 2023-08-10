import math
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.figure(figsize=(12,12))

# Generate a default Gaussian
g = [np.random.normal(0, 1) for x in range(10000)]
plt.hist(g, 200, range=(-5,5), normed=True, facecolor='green', alpha=0.2, histtype='stepfilled')

plt.show()

plt.figure(figsize=(12,12))

# Generate the base Gaussian
g = [np.random.normal(0, 1) for x in range(10000)]

# Generate the square of each element
c = [x**2 for x in g]
plt.hist(c, 200, range=(-5,5), normed=True, facecolor='red', alpha=0.2, histtype='stepfilled')

plt.show()

plt.figure(figsize=(12,12))

# Generate Chi squared based on library function
c = [np.random.chisquare(1) for x in range(10000)]
plt.hist(c, 200, range=(-5,5), normed=True, facecolor='red', alpha=0.2, histtype='stepfilled')

plt.show()

plt.figure(figsize=(12,12))

# Generate Chi squared based on library function
c = [np.random.chisquare(1) for x in range(10000)]

# Generate the values and edges of a histogram
v, e = np.histogram(c, bins=1000, range=(0,10), density=True)

# Generate cumulative sum of the values
v = np.cumsum(v)

# Generate support set from the edges (to plot)
sup = [(e[i] + e[i+1]) / 2 for i in range(len(e)-1)]

# Plot the CDF
plt.plot(sup, v, color='green', alpha=0.5)  
plt.show()

# Generate Chi squared based on library function
c = [np.random.chisquare(1) for x in range(1000000)]

# Generate the values and edges of a histogram
v, e = np.histogram(c, bins=1000, range=(0,10), density=True)

# Generate cumulative sum of the values
v = np.cumsum(v) / 100

# Generate support set from the edges (to plot)
sup = [(e[i] + e[i+1]) / 2 for i in range(len(e)-1)]

# Zip the lists and print them
l = zip(sup, v)
    
# Fetch values within a small tolerance of what you might be looking for
val = 0.95
tol = 0.001
res = []
for x in l:
    if (math.fabs(x[1] - val) < tol):
        print(x)



