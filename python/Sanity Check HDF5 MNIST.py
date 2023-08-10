from neon.data import HDF5Iterator  # Neon's HDF5 data loader
from neon.backends import gen_backend
import numpy as np

be = gen_backend(backend='cpu', batch_size=1)  

outFilename = 'mnist_test.h5'  # The name of our HDF5 data file

train_set = HDF5Iterator(outFilename)

train_set.get_description()

train_set.ndata   # Number of patients in our dataset

train_set.lshape   # DICOM image tensor (C x H x W x D)

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

plt.imshow(train_set.inp[0,:].reshape(28,28), cmap=cm.gray);

i = 0
for x, t in train_set:
    
    print(i)
    print(x)
    plt.imshow(x.get().reshape(28,28), cmap=cm.gray); 
    
    i += 1
    break



