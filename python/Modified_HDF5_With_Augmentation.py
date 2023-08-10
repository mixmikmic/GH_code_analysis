from neon.data import HDF5Iterator  # Neon's HDF5 data loader
from neon.backends import gen_backend

import itertools

be = gen_backend(backend='cpu', batch_size=24)  

#train_set = HDF5Iterator('/Users/ganthony/Desktop/luna16_roi_subset0_augmented.h5')
#train_set_flipped = HDF5Iterator('/Users/ganthony/Desktop/luna16_roi_subset0_augmented.h5', 
#                                 flip_enable=False, rot90_enable=False, crop_enable=True, border_size=7)
train_set = HDF5Iterator('/Users/ganthony/Desktop/mnist_test.h5')
train_set_flipped = HDF5Iterator('/Users/ganthony/Desktop/mnist_test.h5', flip_enable=True, rot90_enable=True, crop_enable=False, border_size=6)

from matplotlib import pyplot as plt
import numpy as np

get_ipython().magic('matplotlib inline')

j = 0
for x,y in itertools.izip(train_set, train_set_flipped):
    
    for i in range(x[0].shape[1]):
        plt.subplot(1,2,1)
        plt.imshow(x[0].get()[:,i].reshape(train_set.lshape)[0], cmap='bone')
        plt.subplot(1,2,2)
        plt.imshow(y[0].get()[:,i].reshape(train_set_flipped.crop_shape)[0], cmap='bone')
        plt.show()
        
    j += 1
    if (j >= 1):
        break



