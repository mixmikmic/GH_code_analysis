from matplotlib import pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np 

import os 

from keras.models import load_model
import h5py
from scipy.stats import pearsonr
# In[48]:

# Load the data to get the number of features
datadir = "/Users/nicolasfarrugia/Documents/recherche/git/Gold-MSI-LSD77/behav"

X = np.load(os.path.join(datadir,"X_y_lsd77_static_tangent.npz"))['X']

n_feat = X.shape[1]

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers, callbacks, regularizers, initializers
from E2E_conv import *
from E2E_conv import *

batch_size = 14
dropout = 0.5
momentum = 0.9
noise_weight = 0.125
lr = 0.01
decay = 0.0005

# Setting l2_norm regularizer
reg = regularizers.l2(decay)
kernel_init = initializers.he_uniform()


model = Sequential()
model.add(E2E_conv(2,32,(2,n_feat),kernel_regularizer=reg,input_shape=(n_feat,n_feat,1),input_dtype='float32',data_format="channels_last"))
print("First layer output shape :"+str(model.output_shape))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(E2E_conv(2,32,(2,n_feat),kernel_regularizer=reg,data_format="channels_last"))
print(model.output_shape)
model.add(LeakyReLU(alpha=0.33))
model.add(Convolution2D(64,(1,n_feat),kernel_regularizer=reg,data_format="channels_last"))
model.add(LeakyReLU(alpha=0.33))
model.add(Convolution2D(256,(n_feat,1),kernel_regularizer=reg,data_format="channels_last"))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(128,kernel_regularizer=reg,kernel_initializer=kernel_init))
#print(model.output_shape)
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(30,kernel_regularizer=reg,kernel_initializer=kernel_init))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(2,kernel_regularizer=reg,kernel_initializer=kernel_init))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.33))
model.summary()

model.load_weights("BrainCNN-gmsi.h5")

y = np.load(os.path.join(datadir,"X_y_lsd77_static_tangent.npz"))['y']
labels = np.load(os.path.join(datadir,"X_y_lsd77_static_tangent.npz"))['labels']
ages = y[:,1]

print(" Selecting only ",labels[3],",",labels[4])


y=y[:,[3,4]]

import numpy as np
from nilearn import datasets
from nilearn.plotting import find_xyz_cut_coords
from nilearn.image import math_img

basc = datasets.fetch_atlas_basc_multiscale_2015(version='sym')['scale064']

import nibabel as nib 

nib_basc444 = nib.load(basc)
labels_data = nib_basc444.get_data()   

#fetch all possible label values 
all_labels = np.unique(labels_data)
# remove the 0. value which correspond to voxels out of ROIs
all_labels = all_labels[1:]


allcoords=[]
for i,curlabel in enumerate(all_labels):
    img_curlab = math_img(formula="img==%d"%curlabel,img=nib_basc444)
    allcoords.append(find_xyz_cut_coords(img_curlab))

import braincnn_vis as bcvis

from importlib import reload 

reload(bcvis)

alpha = 1e-7
niter = 1000

heatmaptest,loss_array,loss_names = bcvis.visualize_activation_ternary_dynamic(model,
                                                verbose = 0,alpha=alpha,
                                                   layer_idx=-1,
                                                   filter_indices=0,
                                                   act_max_weight=1, 
                                                   lp_norm_weight=0,
                                                   tv_weight=0,max_iter=niter)

print("Min Loss Ternary : %f" % np.min(loss_array[:,1]) )

print("Min Loss Activation : %f" % np.min(loss_array[:,0]) )

plt.subplot(3,1,1)
plt.plot(loss_array[:,0])
plt.legend([loss_names[0]])

alphamult = [alpha]
for i in range(niter-1):
    alphamult.append(alphamult[-1]*1.03)
    
    
loss_array[:,1] = loss_array[:,1] / alphamult

plt.subplot(3,1,2)
plt.semilogy(loss_array[:,1])
plt.legend([loss_names[1]])

plt.subplot(3,1,3)
plt.semilogy(alphamult)
plt.legend(['Alpha'])
plt.show()

plt.hist(heatmaptest[:,:,0].ravel())

plt.imshow(((heatmaptest[:,:,0])),interpolation='nearest',cmap=plt.cm.RdBu_r)
plt.colorbar()

from nilearn.plotting import plot_connectome

plot_connectome((heatmaptest[:,:,0]),allcoords,edge_threshold = '95%')





