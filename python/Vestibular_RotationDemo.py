get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import skinematics as skin

t = np.arange(0,10,0.1)
gyr = np.vstack( (np.zeros_like(t), 0.1*np.sin(t), np.zeros_like(t)) ).T
plt.plot(gyr)
plt.legend(['x', 'y', 'z'])

Rmat = skin.rotmat.R1(-5)
#print(Rmat)

gyrRotated = Rmat.dot(gyr.T).T
# If you are running Python 3.5, you can instead write
# gyrRotated = (Rmat @ gyr.T).T

plt.plot(gyrRotated)
plt.legend(['x', 'y', 'z'])

R = np.matrix(Rmat)
omega = np.matrix(gyr)
rotated = (R * gyr.T).T
plt.plot(rotated)

q = [-np.sin(np.deg2rad(5)/2), 0, 0]
rotated = skin.vector.rotate_vector(gyr, q)
plt.plot(rotated)
    



