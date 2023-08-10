# import numpy as np
import time
import os
import h5py
import glob
import IPython.display
import matplotlib.pyplot as plt
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Merge
from keras.layers import Convolution2D, MaxPooling2D
from keras import optimizers
from keras.layers.advanced_activations import ELU
from keras.utils import np_utils
from keras import backend as K
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
from PIL import Image
from keras.layers.normalization import BatchNormalization

# Define the parameters for training
batch_size = 128
nb_classes = 2
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 9, 9

# Volume of the training set
sample_number = 100000

# number of conv filters to use
nb_filters = 112

# CNN kernel size
kernel_size = (3,3)

# load the patches
X1_train = np.zeros((sample_number, img_rows, img_cols))
X2_train = np.zeros((sample_number, img_rows, img_cols))
y_train = np.zeros((sample_number,))

tic = time.time()
# Load the training set.
hdf5TrainPatchesPath = "/home/qw2208/research/trainPatchesOne.hdf5"
with h5py.File(hdf5TrainPatchesPath, "r") as f1:
    for i in xrange(sample_number/2):
        X1_train[2*i,:,:] = f1['left/'+str(i)][()]
        X1_train[(2*i+1),:,:] = f1['left/'+str(i)][()]
        X2_train[2*i,:,:] = f1['rightNeg/'+str(i)][()]
        X2_train[(2*i+1),:,:] = f1['rightPos/'+str(i)][()]
        y_train[2*i] = 0
        y_train[2*i+1] = 1
    
toc = time.time()
print "Time for loading the training set: ", toc-tic

# Resize the dataset (Trivial)
if K.image_dim_ordering() == 'th':
    X1_train = X1_train.reshape(X1_train.shape[0], 1, img_rows, img_cols)
    X2_train = X2_train.reshape(X2_train.shape[0], 1, img_rows, img_cols)
    input_shape = (1,img_rows, img_cols)
else:
    X1_train = X1_train.reshape(X1_train.shape[0], img_rows, img_cols, 1)
    X2_train = X2_train.reshape(X2_train.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols,1)

# for i in xrange(20,30):
#     print 'Check {}'.format(i-19)
#     print X1_train[2*i][0]
#     # print (X1_train[2*i+1][0]-X2_train[2*i+1][0])

X1_train = X1_train.astype('float32')
X2_train = X2_train.astype('float32')

for i in xrange(3):
    y_train = np.expand_dims(y_train,axis=2)
print y_train.shape

# https://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/#merge
left_branch = Sequential()
left_branch.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same', input_shape=input_shape))
#left_branch.add(BatchNormalization(axis=1))
left_branch.add(ELU())
left_branch.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
#left_branch.add(BatchNormalization(axis=1))
left_branch.add(ELU())
left_branch.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
#left_branch.add(BatchNormalization(axis=1))
left_branch.add(ELU())
left_branch.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
#left_branch.add(BatchNormalization(axis=1))
left_branch.add(ELU())

right_branch = Sequential()
right_branch.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same', input_shape=input_shape))
#right_branch.add(BatchNormalization(axis=1))
right_branch.add(ELU())
right_branch.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
#right_branch.add(BatchNormalization(axis=1))
right_branch.add(ELU())
right_branch.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
#right_branch.add(BatchNormalization(axis=1))
right_branch.add(ELU())
right_branch.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
#right_branch.add(BatchNormalization(axis=1))
right_branch.add(ELU()) 

merged = Merge([left_branch, right_branch], mode='concat', concat_axis=1)
nb_filters_fc = 384

fc = Sequential()
fc.add(merged)
fc.add(Convolution2D(nb_filters_fc, img_rows, img_cols, border_mode='valid'))
#fc.add(BatchNormalization(axis=1))
fc.add(ELU())
fc.add(Convolution2D(nb_filters_fc, 1,1, border_mode='valid'))
#fc.add(BatchNormalization(axis=1))
fc.add(ELU())
fc.add(Convolution2D(nb_filters_fc, 1,1, border_mode='valid'))
#fc.add(BatchNormalization(axis=1))
fc.add(ELU())
fc.add(Convolution2D(1, 1, 1, border_mode='valid', activation='sigmoid'))

optimizer = optimizers.RMSprop(lr=1e-4, decay=1e-5)
# fc.load_weights('/home/qw2208/research/weightsOne.hdf5')
fc.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
fc.fit([X1_train,X2_train], y_train, validation_split=0.1, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=True, callbacks=[TQDMNotebookCallback()])
# Evaluate the result based on the training set
score = fc.evaluate([X1_train,X2_train], y_train, verbose=0)
# print score.shape
print('Test score: ', score[0])
print('Test accuracy: ', score[1])

fc.summary()

fc.save('/home/qw2208/research/weightsOne.hdf5')



