get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from math import *
import matplotlib.patches as patches
import matplotlib.image as mpimg
img=mpimg.imread('track1_top_view.png')

plt.rcParams["figure.figsize"] = [11, 11]

waypoint_file = "lake_track_waypoints.csv"
ml_waypoint_file = "lake_track_ml_generated_waypoints.csv"

with open(waypoint_file) as f:
    x_waypoint = []
    y_waypoint = []
    count = 0
    for line in f:
        if count > 0:
            data = line.split(',')
            x_waypoint.append(data[0])
            y_waypoint.append(data[1])
        count += 1
    
with open(ml_waypoint_file) as f:
    x_ml_waypoint = []
    y_ml_waypoint = []
    count = 0
    for line in f:
        if count > 0:
            data = line.split(',')
            x_ml_waypoint.append(data[0])
            y_ml_waypoint.append(data[1])   
        count += 1
        
x_start = [ x_ml_waypoint[0] ]
y_start = [ y_ml_waypoint[1] ]

#table_waypoints

p0 = plt.imshow(img, extent=(-338, 315, -213, 203))
p2 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p3 = plt.plot(x_start, y_start, 'ro', ms=5.0)
plt.title('Track 1 Map')
plt.legend((p2[0], p3[0]), ('Provided Waypoints', 'Starting Position'))
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203))
p1 = plt.plot(x_ml_waypoint, y_ml_waypoint, 'b', ms=0.1)
p2 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p3 = plt.plot(x_start, y_start, 'ro', ms=5.0)
plt.title('Track 1 Map')
plt.legend((p1[0], p2[0], p3[0]), ('ML Generated Waypoints', 'Provided Waypoints', 'Starting Position'))
plt.show()



