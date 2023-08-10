get_ipython().magic('matplotlib inline')

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling3D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from scipy.misc import imresize 

import matplotlib.pylab as plt

from lfw_fuel import lfw

# !!!!!!!!!!!!!!!!
# Be patient!
# This step takes about 2 minutes
# !!!!!!!!!!!!!!!!
# 
# Load the data, shuffled and split between train and test sets
(X_train_original, y_train_original), (X_test_original, y_test_original) = lfw.load_data("deepfunneled")

print(X_train_original.shape)
print(X_test_original.shape)

im = X_train_original[51,:,:,:]
print(im[0,:,:])

fig = plt.figure()
[ax1, ax2] = [fig.add_subplot(1,2,i+1) for i in range(2)]
ax1.imshow(im[0,:,:])
ax2.imshow(im[3,:,:])
plt.show()

print(y_train_original[51])

# Original images are 250 x 250
print(im.shape)

# Crop the image to 128 x 128

# Do a little margin math
current_dim = 250
target_dim = 128
margin = int((current_dim - target_dim)/2)
left_margin = margin
right_margin = current_dim - margin

newim = im[:, left_margin:right_margin, left_margin:right_margin]
print(newim.shape)

# This transpose is mainly useful for plotting with color:
# Put the images in standard dimension order
# (width, height, channels)
sized1 = newim[0:3,:,:]
sized1 = np.transpose(sized1,(1,2,0))

sized2 = newim[3:6,:,:]
sized2 = np.transpose(sized2,(1,2,0))

fig = plt.figure()
[ax1, ax2] = [fig.add_subplot(1,2,i+1) for i in range(2)]

ax1.imshow(sized1)
ax2.imshow(sized2)
plt.show()

feature_width = feature_height = 32

resized1_32 = imresize(newim[0:3,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")
resized2_32 = imresize(newim[3:6,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")
print(resized1_32.shape)
print(resized2_32.shape)

feature_width = feature_height = 64

resized1_64 = imresize(newim[0:3,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")
resized2_64 = imresize(newim[3:6,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")
print(resized1_64.shape)
print(resized2_64.shape)

fig = plt.figure(figsize=(6,12))
[ax1, ax2, ax3, ax4, ax5, ax6] = [fig.add_subplot(3,2,i+1) for i in range(6)]

ax1.imshow(sized1)
ax2.imshow(sized2)
ax3.imshow(resized1_64)
ax4.imshow(resized2_64)
ax5.imshow(resized1_32)
ax6.imshow(resized2_32)
plt.show()

print(X_train_original[0,:,:,:].shape)

current_dim = 250
target_dim = 128
margin = int((current_dim - target_dim)/2)
left_margin = margin
right_margin = current_dim - margin

# newim is shape (6, 128, 128)
newim = im[:, left_margin:right_margin, left_margin:right_margin]

# resized are shape (feature_width, feature_height, 3)
feature_width = feature_height = 32
resized1 = imresize(newim[0:3,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")
resized2 = imresize(newim[3:6,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")

# re-packge into a new X entry
newX = np.concatenate([resized1,resized2], axis=2)

fig = plt.figure()
ax1, ax2 = [fig.add_subplot(1,2,i+1) for i in range(2)]
ax1.imshow(newX[:,:,0:3])
ax2.imshow(newX[:,:,3:6])

def crop_and_downsample(originalX, downsample_size=32):
    """
    Starts with a 250 x 250 image.
    Crops to 128 x 128 around the center.
    Downsamples the image to (downsample_size) x (downsample_size).
    Returns an image with dimensions (channel, width, height).
    """
    current_dim = 250
    target_dim = 128
    margin = int((current_dim - target_dim)/2)
    left_margin = margin
    right_margin = current_dim - margin

    # newim is shape (6, 128, 128)
    newim = im[:, left_margin:right_margin, left_margin:right_margin]

    # resized are shape (feature_width, feature_height, 3)
    feature_width = feature_height = downsample_size
    resized1 = imresize(newim[0:3,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")
    resized2 = imresize(newim[3:6,:,:], (feature_width, feature_height), interp="bicubic", mode="RGB")

    # re-packge into a new X entry
    newX = np.concatenate([resized1,resized2], axis=2)
    return newX

X_subsample_original = X_train_original[50:60,:,:,:]
X_subsample_transformed = np.asarray([crop_and_downsample(x) for x in X_subsample_original])

print(X_subsample_original.shape)
print(X_subsample_transformed.shape)

fig = plt.figure()
ax1, ax2 = [fig.add_subplot(1,2,i+1) for i in range(2)]

ax1.imshow(X_subsample_transformed[1,:,:,0:3])
ax2.imshow(X_subsample_transformed[1,:,:,3:6])

