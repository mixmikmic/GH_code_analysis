import random # Random number generation
import matplotlib.pyplot as plt  # Plotting
from math import pi, pow, sqrt  # Math functions

def distance(a, b):
    return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))

NUMBER_OF_POINTS = 10000 # Change this value to see how the approximation changes!

points = ((random.uniform(0, 1), random.uniform(0, 1)) for i in range(NUMBER_OF_POINTS))

inside_quadrant = 0
outside_quadrant = 0

plt.axis('equal') # Ensure plot is a box

for p in points:
    dist = distance((0, 0), p)  # Distance from origin
    if dist < 1:
        inside_quadrant += 1
        plt.scatter(x=p[0], y=p[1], color="#3F51B5", alpha=0.5) # Blue points
    else:
        outside_quadrant += 1
        plt.scatter(x=p[0], y=p[1], color="#F44336", alpha=0.5) # Red points
        
plt.show() # Show the plot

ratio = inside_quadrant / (inside_quadrant + outside_quadrant)
approximation = ratio * 4
accuracy = (approximation / pi * 100) if (approximation <= pi) else (pi / approximation * 100)

print("𝛑 is approximately {}".format(approximation))
print("This approximation is {}% accurate".format(accuracy))

