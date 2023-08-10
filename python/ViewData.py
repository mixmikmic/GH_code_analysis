import time
import random
import numpy as np
from collections import defaultdict
from optparse import OptionParser

# Required libraries
import h5py
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization as BN

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

import sys
sys.path.append('../repo/d-script/')
# d-script imports
from data_iters.minibatcher import MiniBatcher
from data_iters.iam_hdf5_iterator import IAM_MiniBatcher

hdf5_file = '/memory/raw_forms_uint8.hdf5'
hdf5_file = '/memory/raw_lines_from_forms_uint8.hdf5'
hdf5_file = '/memory/raw_words_uint8.hdf5'
num_authors=5
num_forms_per_author=-1
shingle_dim=(120,1909)
shingle_dim=(100,200)
use_form=True
batch_size=32

iam_m = IAM_MiniBatcher(hdf5_file, num_authors, num_forms_per_author, shingle_dim=shingle_dim, use_form=use_form, default_mode=MiniBatcher.TRAIN, batch_size=batch_size)

[X_test, Y_test] = iam_m.get_train_batch(batch_size*20)
print "Shape="+str(X_test.shape)+", Max="+str(X_test.max())

plt.clf
plt.subplots(3,3)
# plt.rcParams['figure.figsize'] = (40.0, 10.0)
# plt.figure(figsize=(20,10))
# fig,axes = plt.rcParams['figure.figsize'] = 60, 60
# plt.tight_layout()
# s = np.random.choice(32,9, replace=False)
s=xrange(9)
for i in xrange(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[s[i]])





