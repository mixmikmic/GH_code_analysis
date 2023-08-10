get_ipython().magic('run plot_normballs.py')

get_ipython().magic('matplotlib inline')
import numpy as np
np.set_printoptions(precision=3, suppress=True)


get_ipython().magic('run matrix_norm_sliders.py')

get_ipython().magic('matplotlib inline')

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from notes_utilities import pnorm_ball_points

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grab some test data.

x = np.arange(-2,2,0.1)

X, Y = np.meshgrid(x, x)

p2 = 2
p = p2
Z2 = (np.abs(X)**p + np.abs(Y)**p)**(1./p) 

p1 = 5
p = p1
Z = (np.abs(X)**p + np.abs(Y)**p)**(1./p) 

r = 2
dx, dy = pnorm_ball_points(p=p1)

ax.plot(r*dx,r*dy,r*np.ones_like(dx), color='b')
ax.plot(r*dx,r*dy,np.sqrt(2.)*r*np.ones_like(dx), color='g')
dx, dy = pnorm_ball_points(p=p2)
ax.plot(r*dx,r*dy,r*np.ones_like(dx), color='r')


# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4)
ax.plot_wireframe(X, Y, Z2, rstride=4, cstride=4, color='r' )
ax.plot_wireframe(X, Y, np.sqrt(2.)*Z, rstride=4, cstride=4, color='g' )

plt.show()

