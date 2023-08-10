get_ipython().magic('reset')

import numpy as np
from numpy import cos, sin, sqrt, arange, pi

# to plot our result
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
mpl.rcParams['figure.figsize'] = (14,10)
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14
#get_ipython().magic('matplotlib inline') # inline plotting

# parameters
dt = 0.01
tmax = 3.0
g = 9.8
m = 0.150
c = 0.02;

timeRange = arange(0, tmax, dt);

# initial conditions
initialAngle = 50 * pi/180;
speed = 30.0
vx = speed * cos (initialAngle)
vy = speed * sin (initialAngle)
x = 0
y = 0

# storing arrays
vxs = []
vys = []

xs = []
ys = []

# loop over time
for t in timeRange:
     ax = - c/m * sqrt( vx**2 +vy**2)*vx
     ay = - g -c/m * sqrt( vx**2 +vy**2)*vy
    
     x = x + vx * dt + 1/2 * ax * dt**2
     y = y + vy * dt + 1/2 * ay * dt**2
    
     vx = vx + ax * dt
     vy = vy + ay * dt

     # storing results 
     vxs.append(vx)
     vys.append(vy)
    
     xs.append(x)
     ys.append(y)
        

# plot the solution 
plt.plot(timeRange, xs)
plt.plot(timeRange, ys)
plt.ylabel('displacement (m)')
plt.xlabel('time (s)')

# trajectory 
plt.plot(xs,ys)
plt.ylabel('y (m)')
plt.xlabel('x (m)')

r= np.vstack([xs,ys])

# maximum height
maxHeight=np.max(ys)
maxHeight

yIndex=ys.index(maxHeight)
yIndex

np.argmax(ys)

# time at maximum height
timeRange[yIndex]

# x position at maximum height
xs[yIndex]

# create numpy arrays
x=np.array(xs)
y=np.array(ys)

xIndex=np.min(np.where(y<0))

timeOfFlight = timeRange[xIndex]
timeOfFlight

xRange = x[xIndex]
xRange

x[xIndex-1]

