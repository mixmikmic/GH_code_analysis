# Import libraries
import math
import numpy as np
import matplotlib.pyplot as plt

in_circle = 0
outside_circle = 0

n = 10 ** 4

# Draw many random points
X = np.random.rand(n)
Y = np.random.rand(n)

for i in range(n):
    
    if X[i]**2 + Y[i]**2 > 1:
        outside_circle += 1
    else:
        in_circle += 1

area_of_quarter_circle = float(in_circle)/(in_circle + outside_circle)
pi_estimate = area_of_circle = area_of_quarter_circle * 4

pi_estimate

# Plot a circle for reference
circle1=plt.Circle((0,0),1,color='r', fill=False, lw=2)
fig = plt.gcf()
fig.gca().add_artist(circle1)
# Set the axis limits so the circle doesn't look skewed
plt.xlim((0, 1.8))
plt.ylim((0, 1.2))
plt.scatter(X, Y)

in_circle = 0
outside_circle = 0

n = 10 ** 3

# Draw many random points
X = np.random.rand(n)
Y = np.random.rand(n)

# Make a new array
pi = np.ndarray(n)

for i in range(n):
    
    if X[i]**2 + Y[i]**2 > 1:
        outside_circle += 1
    else:
        in_circle += 1

    area_of_quarter_circle = float(in_circle)/(in_circle + outside_circle)
    pi_estimate = area_of_circle = area_of_quarter_circle * 4
    
    pi[i] = pi_estimate
    
plt.plot(range(n), pi)
plt.xlabel('n')
plt.ylabel('pi estimate')
plt.plot(range(n), [math.pi] * n)

