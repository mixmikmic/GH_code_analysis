import numpy as np
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
from keras.layers.advanced_activations import ELU
from keras.utils import np_utils
from keras import backend as K
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
from PIL import Image
from keras.layers.normalization import BatchNormalization

# Read test images from local host
X1 = np.array(Image.open("/home/qw2208/research/left1.png"))
X1 = (X1-np.mean(X1))/np.std(X1)
X2 = np.array(Image.open("/home/qw2208/research/right1.png"))
X2 = (X2-np.mean(X2))/np.std(X2)
# input image dimensions
img_rows, img_cols = X1.shape[0], X1.shape[1]
input_shape = (1, img_rows, img_cols)

X1 = X1.reshape(1, 1, img_rows, img_cols)
X2 = X2.reshape(1, 1, img_rows, img_cols)

# number of conv filters to use
nb_filters = 112

# CNN kernel size
kernel_size = (3,3)

X1 = X1.astype('float32')
X2 = X2.astype('float32')

# Define CNN
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

cnn = Sequential()
cnn.add(merged)

def load_cnn_weights(model, filepath):
    f = h5py.File(filepath, mode='r')
    # g = f['model_weights']
    # print f["conv2d_1/conv2d_1"]
    weights = []
    for i in range(1, 9):
        weights.append(f['model_weights/convolution2d_{}/convolution2d_{}_W/'.format(i, i)][()])
        weights.append(f['model_weights/convolution2d_{}/convolution2d_{}_b/'.format(i, i)][()])
#         weights.append(f['model_weights/batchnormalization_{}/batchnormalization_{}_beta/'.format(i-2, i-2)][()])
#         weights.append(f['model_weights/batchnormalization_{}/batchnormalization_{}_gamma/'.format(i-2, i-2)][()])
#         weights.append(f['model_weights/batchnormalization_{}/batchnormalization_{}_running_mean/'.format(i-2, i-2)][()])
#         weights.append(f['model_weights/batchnormalization_{}/batchnormalization_{}_running_std/'.format(i-2, i-2)][()])
    # print len(weights)
    model.set_weights(weights)
    f.close()
    
# load weight for first cnn part
load_cnn_weights(cnn, "/home/qw2208/research/weightsOne.hdf5")

# predict feature map output and later will do d times fc
output_cnn = cnn.predict([X1, X2])
print "output shape is =====================>", output_cnn.shape
print (output_cnn)

# set network params for fc
nb_filters_fc = 384
kernel_size = (9, 9) 
input_shape = (nb_filters*2, None, None)

def load_fc_weights(filepath):
    f = h5py.File(filepath, mode='r')
    weights = []
    for i in range(9, 13):
        weights.append(f['model_weights/convolution2d_{}/convolution2d_{}_W/'.format(i, i)][()])
        weights.append(f['model_weights/convolution2d_{}/convolution2d_{}_b/'.format(i, i)][()])
#         weights.append(f['model_weights/batchnormalization_{}/batchnormalization_{}_beta/'.format(i-2, i-2)][()])
#         weights.append(f['model_weights/batchnormalization_{}/batchnormalization_{}_gamma/'.format(i-2, i-2)][()])
#         weights.append(f['model_weights/batchnormalization_{}/batchnormalization_{}_running_mean/'.format(i-2, i-2)][()])
#         weights.append(f['model_weights/batchnormalization_{}/batchnormalization_{}_running_std/'.format(i-2, i-2)][()])
    return weights

weights_fc = load_fc_weights("/home/qw2208/research/weightsOne.hdf5")

# create original fully-connected layers for training but now fully-conv layers

fc = Sequential()
fc.add(Convolution2D(nb_filters_fc, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
# fc.add(BatchNormalization(axis=1))
fc.add(ELU())
fc.add(Convolution2D(nb_filters_fc, 1, 1, border_mode='valid'))
# fc.add(BatchNormalization(axis=1))
fc.add(ELU())
fc.add(Convolution2D(nb_filters_fc, 1, 1, border_mode='valid'))
# fc.add(BatchNormalization(axis=1))
fc.add(ELU())
fc.add(Convolution2D(1, 1, 1, border_mode='valid', activation='sigmoid'))

fc.set_weights(weights_fc)

# input feature map into fully-conv(test phase) layer for d times
d_max = 200
vol = np.zeros((img_rows-8, img_cols-8, d_max), dtype=np.float)
for d in range(1, d_max+1):
    input_fc_left = output_cnn[:, 0:112, :, d:]
    input_fc_right = output_cnn[:, 112:, :, 0:-d]
    input_fc = np.concatenate((input_fc_left, input_fc_right), axis=1)
    
    print input_fc.shape
    output = fc.predict(input_fc)
    print output
    vol[:, d:, d-1] = output.squeeze()
#    print "============================= ", d

result_index = np.argmax(vol, axis=2)
print result_index.shape, "\n ", result_index
result_index = result_index.astype('int16')
im = Image.fromarray(result_index)
im.convert('RGB').save("disp.png")
print "Finished!"



