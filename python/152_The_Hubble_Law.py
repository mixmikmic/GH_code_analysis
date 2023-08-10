get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from matplotlib import pylab
from numpy import arange,array,ones
from scipy import stats

q1_distances = [0.45, 0.9, 0.9, 1.7, 2.0, 2.0, 2.0]
q1_speeds = [200, 650, 500, 960, 850, 800, 1090]

plt.scatter(q1_distances, q1_speeds)
plt.xlabel("Distance (mpc)")
plt.ylabel("Speed (km/s)")
plt.title("Hubble's Law", fontsize=20)
plt.grid()
plt.show()

q2_max_distance = max(q1_distances)* 1000000 * 3.26 
q2_min_distance = min(q1_distances)* 1000000 * 3.26

print("The minimum distance is", q2_min_distance, 
      "light years, \nthe maximum distance is", q2_max_distance, 
      "light years, \nand the range is ", q2_max_distance - q2_min_distance, 
      "light years.")

slope, intercept, r_value, p_value, std_err = stats.linregress(q1_distances, q1_speeds)
line = []

for i in q1_distances:
    line.append(slope * i + intercept)
    
plt.plot(q1_distances,q1_speeds,'o', q1_distances, line)
pylab.title("Hubble's Law - Linear Regression")
plt.grid()

print("The regression has a slope of", slope, "kilometers/sec per mpc.")

def q6_distance(speed):
    '''Calculates distance in megaparsecs when given speed in kilometers
    per second, using the regression equation.'''
    distance = (speed - intercept) / slope
    return distance

q6_distance(2500)

