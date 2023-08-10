get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import random
import math

f = lambda x: 5 * np.sin(6 * x) + 3 * np.sin(2 * x) + 7
x = np.linspace(0, 10, 1000)
y = f(x)

_ = plt.plot(x,y)

NUM_POINTS = 10000
rect_width = 10
rect_height = 14

rand_x = lambda: random.uniform(0, rect_width)
rand_y = lambda: random.uniform(0, rect_height)

points = [(rand_x(), rand_y()) for i in range(NUM_POINTS)]
points_under = [point for point in points if point[1] <= f(point[0])]
points_above = list(set(points) - set(points_under))

# Separate x's and y's to pass to scatter function.
(under_x, under_y) = zip(*list(points_under))
(over_x, over_y) = zip(*list(points_above))

fig = plt.figure()
fig.set_size_inches(12, 8)
_ = plt.scatter(under_x, under_y, s=1, color='red')
_ = plt.scatter(over_x, over_y, s=1, color='green')

# Area = area of domain rectangle * num_points_under/num_points_total
area = rect_width * rect_height * len(points_under)*1.0/len(points)
print("Estimate of area under the curve:", area)

# Sanity check: it looks like the area under is about half of the rectangle, and the rectangle
# area is 10*14 = 140, so it should be around 70.

import random

NUM_POINTS = 10000

# Randomly generate points (x[i], y[i]) such that -1 <= x[i] = 1 and -1 <= y[i] <= 1.
x = [random.uniform(-1,1) for i in range(NUM_POINTS)]
y = [random.uniform(-1,1) for i in range(NUM_POINTS)]

circle_x = []
circle_y = []

outsiders_x = []
outsiders_y = []

# Determine which points are inside the circle (and for visualization purposes, also
# determine which are outside the circle).
for i in range(NUM_POINTS):
    if x[i]**2 + y[i]**2 <= 1:
        circle_x.append(x[i])
        circle_y.append(y[i])
    else:
        outsiders_x.append(x[i])
        outsiders_y.append(y[i])

# Plot it.
fig = plt.figure()
fig.set_size_inches(10, 10)
_ = plt.scatter(outsiders_x, outsiders_y, s=1, color='green')
_ = plt.scatter(circle_x, circle_y, s=1, color='red')

print("Estimate of area of circle (pi):", 4 * (len(circle_x)*1.0 / len(x)))



