import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
plt.style.use('notebook');
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import matplotlib.patches as patches
from scipy.constants import g,pi
π = pi

# drag coeeficient divided by mass
B2oMass = 4.0E-5 # 1/m

# the time step
Δt = 0.01 # s

# initial velocity
v0 = 500 # m/s

# guess for angle
θ = np.radians(20.0) # radians

# angle step
dθ = 0.01 # radians

# Iterate the Euler equations for the positions, only storing the x and y
# coordinates until we have a 'hit'
# hint: check out the docs for numpy.append and consider using while loops

xmin,xmax = 13.5E3,13.55E3
x = np.array([0.0])
while not (xmin < x[-1] < xmax):
    x = np.array([0.0])
    y = np.array([0.0])
    θ += dθ
    vx = v0 * np.cos(θ) 
    vy = v0 * np.sin(θ) 

    while y[-1] >= 0.0:
        v = np.sqrt(vx**2 + vy**2)
        vx -= B2oMass * v * vx * Δt
        vy -= g*Δt + B2oMass * v * vy * Δt
        x = np.append(x,x[-1] + vx*Δt)
        y = np.append(y,y[-1] + vy*Δt)
        
    if x[-1] > xmax:
        print('Too Far!')
        break

# Plot the resulting trajectory
ax1 = plt.subplot(111)
ax1.plot(x/1.0E3,y/1.0E3,'-',label='θ = %4.2f˚'%np.degrees(θ))
ax1.add_patch(patches.Rectangle((13.5, 0.0),0.05,0.05,facecolor='#39FF14',edgecolor='None',zorder=10))

# set the x and y labels and a title
plt.xlabel('x [km]')
plt.ylabel('y [km]')
plt.legend()
plt.grid(True)
plt.title('Cannon Shell Trajectory')

# Only show positive y-coordinates
plt.axis(ymin=0.0);

