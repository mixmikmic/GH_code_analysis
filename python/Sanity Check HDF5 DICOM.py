from neon.data import HDF5Iterator  # Neon's HDF5 data loader
from neon.backends import gen_backend

be = gen_backend(backend='cpu', batch_size=1)  

outFilename = 'dicom_out.h5'  # The name of our HDF5 data file

train_set = HDF5Iterator(outFilename)

train_set.get_description()

train_set.ndata   # Number of patients in our dataset

train_set.lshape   # DICOM image tensor (C x H x W x D)

from matplotlib import pyplot as plt, cm

get_ipython().magic('matplotlib inline')

i = 0
plt.figure(figsize=(40,40))

for x, t in train_set:
    
    print(x)
    plt.subplot(train_set.ndata,1,i+1)
    
    # Print out slice #100 for eah patient
    plt.title('Patient #{}, Slice #{}'.format(i, 100))
    plt.imshow(x.get().reshape(512,512,128)[:,:,100], cmap=cm.bone); 
    
    i += 1

from ipywidgets import interact

def displaySlice(sliceNo):
    plt.figure(figsize=[10,10]);
    plt.title('Patient #0, Slice #{}'.format(sliceNo));
    plt.imshow(train_set.inp[0,:].reshape(512,512,128)[:,:,sliceNo-1], cmap=cm.bone); 
    plt.show()
    
interact(displaySlice, sliceNo=(1,128,1));



