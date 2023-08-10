import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from menpo.shape import PointCloud
import menpo.io as mio
from menpofit.transform import DifferentiableThinPlateSplines

src_landmarks = PointCloud(np.array([[-1, -1],
                                     [-1,  1],
                                     [ 1, -1],
                                     [ 1,  1]]))

tgt_landmarks = PointCloud(np.array([[-1, -1],
                                     [-1,  1],
                                     [ 1, -1],
                                     [ 1,  1]]))

tps = DifferentiableThinPlateSplines(src_landmarks, tgt_landmarks)
np.allclose(tps.apply(src_landmarks).points, tgt_landmarks.points)

x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)
xx, yy = np.meshgrid(x, y)
points = np.array([xx.flatten(1), yy.flatten(1)]).T

get_ipython().magic('matplotlib inline')
dW_dxy = tps.d_dl(points)
reshaped = dW_dxy.reshape(xx.shape + (4,2))

#dW_dx
plt.subplot(241)
plt.imshow(reshaped[:,:,0,0])
plt.subplot(242)
plt.imshow(reshaped[:,:,1,0])
plt.subplot(243)
plt.imshow(reshaped[:,:,2,0])
plt.subplot(244)
plt.imshow(reshaped[:,:,3,0])

#dW_dy
plt.subplot(245)
plt.imshow(reshaped[:,:,0,1])
plt.subplot(246)
plt.imshow(reshaped[:,:,1,1])
plt.subplot(247)
plt.imshow(reshaped[:,:,2,1])
plt.subplot(248)
plt.imshow(reshaped[:,:,3,1])

print(reshaped[1:5,1:5,0,0])
print(reshaped[1:5,1:5,0,1])

summed_x = np.sum(reshaped[:,:,:,0], axis=-1)
np.allclose(np.ones(xx.shape), summed_x)

plt.imshow(summed_x)

summed_y = np.sum(reshaped[:,:,:,1], axis=-1)
np.allclose(np.ones(xx.shape), summed_y)

plt.imshow(summed_y)

np.allclose(reshaped[:,:,:,0], reshaped[:,:,:,1])

