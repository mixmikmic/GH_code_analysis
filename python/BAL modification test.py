get_ipython().magic('matplotlib inline')

import numpy as np
import scipy.ndimage as scnd

import matplotlib.pyplot as plt

np.random.seed(1)


nx = 401
ny = 401
dx = 25.
dy = 25.
ht = np.zeros((nx, ny))

hill_width=2000 # Hill width (m) - same as hill_width in WRF emles code!
idx = hill_width/dx
xs = nx/2 - idx
xe = xs + 2.*idx
ys = ny/2 - idx
ye = ys + 2.*idx
xx, yy = np.meshgrid(np.arange(ny), np.arange(nx))
xm = xx * dx
ym = yy * dx

for i in range(nx):
    for j in range(int(ys), int(ye)):
        ht[i,j] =  250. * 0.5 * (1. + np.cos(2*np.pi/(ye-ys) *(j-ys)+ np.pi))
        
r = np.random.normal(0, 0.5,size=(nx, ny))

ht += r
f = np.ones((3,3))/9.
ht = scnd.convolve(ht, f)

plt.contourf(xm, ym, ht, cmap='terrain')
plt.axes().set_aspect('equal')
plt.colorbar()
plt.title("Elevation (m)")

slope_x, slope_y = np.gradient(ht, dx)
slope_np = np.hypot(slope_x, slope_y)
print(slope_y.max())
print(slope_y.min())


# In[11]:

plt.contourf(xm, ym, slope_np, cmap='seismic')
plt.axes().set_aspect('equal')
plt.colorbar()
plt.title("Hill slope (fractional)")

dzdx = scnd.sobel(ht, axis=1)/(8.*dx)
dzdy = scnd.sobel(ht, axis=0)/(8.*dy)
slope_sc = np.hypot(dzdx, dzdy)

plt.contourf(xm, ym, slope_sc, cmap='seismic')
plt.axes().set_aspect('equal')
plt.colorbar()
plt.title("Hill slope (fractional)")

plt.contourf(xm, ym, slope_sc - slope_np, cmap='seismic')
plt.axes().set_aspect('equal')
plt.colorbar()
plt.title("Hill slope difference")

plt.contourf(xm, ym, (slope_sc - slope_np)/slope_sc, cmap='seismic')
plt.axes().set_aspect('equal')
plt.colorbar()
plt.title("Hill slope difference (fractional)")

import numexpr
RADIANS_PER_DEGREE = np.pi/180.
aspect_array = numexpr.evaluate(
        "(450 - arctan2(dzdy, -dzdx) / RADIANS_PER_DEGREE) % 360")

aspect_np = np.mod((450. - np.arctan2(dzdy, -dzdx)/RADIANS_PER_DEGREE), 360.)


plt.contourf(xm, ym, (aspect_array - aspect_np), cmap='seismic')
plt.axes().set_aspect('equal')
plt.colorbar()
plt.title("Hill aspect difference")

