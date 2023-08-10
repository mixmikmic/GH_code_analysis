# Front matter: load libraries needed for training
import numpy as np
import tensorflow as tf
from utils import *

import os

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Runfile preliminaries
networkSizes = ['Baseline', 'Wide','Deep']
networkTypes = ['Real', 'Complex','SplitComplex']

# Initialize MNIST Data

dataset = 'MNIST' #set book-keeping parameter

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

# Zero-pad MNIST images to be 32x32
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))

# Set training parameters
EPOCHS = 40
BATCH_SIZE = 200
NUM_TRAIN = X_train.shape[0] # set number of training examples
NUM_VAL = X_validation.shape[0] 
NUM_TEST = X_test.shape[0] 

# Loop over sizes and types to generate data

for networkSize in networkSizes:
    for networkType in networkTypes:
        runmodel((X_train,y_train,NUM_TRAIN), (X_validation,y_validation,NUM_VAL), (X_test,y_test,NUM_TEST), 
                 dataset, BATCH_SIZE, EPOCHS, num_classes = 10, networkSize = networkSize, networkType = networkType, alpha = 1e-3)

# Initialize CIFAR-10 Data
# Used code from: https://github.com/tflearn/tflearn/blob/master/tflearn/datasets/cifar10.py
import cifar10
dataset = 'CIFAR10' #set book-keeping parameter

train_tup, test_tup = cifar10.load_data()

X_train, y_train = train_tup
X_validation = X_train[49000:,:]
y_validation = y_train[49000:]
X_train = X_train[:49000,:]
y_train = y_train[:49000]

X_test, y_test = test_tup

# Set training parameters
EPOCHS = 100
BATCH_SIZE = 200
NUM_TRAIN = X_train.shape[0] # set number of training examples
NUM_VAL = X_validation.shape[0] 
NUM_TEST = X_test.shape[0] 
print('Number of training examples: {}'.format(NUM_TRAIN))
print('Number of validation examples: {}'.format(NUM_VAL))
print('Number of testing examples: {}'.format(NUM_TEST))

# Runfile preliminaries
networkSizes = ['Baseline', 'Wide','Deep']
networkTypes = ['Real', 'Complex','SplitComplex']

# Loop over sizes and types to generate data

for networkSize in networkSizes:
    for networkType in networkTypes:
        runmodel((X_train,y_train,NUM_TRAIN), (X_validation,y_validation,NUM_VAL), (X_test,y_test,NUM_TEST), dataset, 
                 BATCH_SIZE, EPOCHS, num_classes = 10, networkSize = networkSize, networkType = networkType, alpha = 1e-4)

# Initialize CIFAR-10 Data
# Used code from: https://github.com/tflearn/tflearn/blob/master/tflearn/datasets/cifar10.py
import cifar10
dataset = 'CIFAR10-L2reg' #set book-keeping parameter

train_tup, test_tup = cifar10.load_data()

X_train, y_train = train_tup
X_validation = X_train[49000:,:]
y_validation = y_train[49000:]
X_train = X_train[:49000,:]
y_train = y_train[:49000]

X_test, y_test = test_tup

# Set training parameters
EPOCHS = 100
BATCH_SIZE = 200
NUM_TRAIN = X_train.shape[0] # set number of training examples
NUM_VAL = X_validation.shape[0] 
NUM_TEST = X_test.shape[0] 

print('Number of training examples: {}'.format(NUM_TRAIN))
print('Number of validation examples: {}'.format(NUM_VAL))
print('Number of testing examples: {}'.format(NUM_TEST))

# Runfile preliminaries
networkSizes = ['Wide','Deep'] # ['Baseline', 'Wide','Deep']
networkTypes = ['Real', 'Complex','SplitComplex']

# Loop over sizes and types to generate data

for networkSize in networkSizes:
    for networkType in networkTypes:
        runmodel((X_train,y_train,NUM_TRAIN), (X_validation,y_validation,NUM_VAL), (X_test,y_test,NUM_TEST), 
                 dataset, BATCH_SIZE, EPOCHS, num_classes = 10, networkSize = networkSize, 
                 networkType = networkType, alpha = 5e-4, lam = 5e-3)

# Initialize CIFAR-100 Data
# Used code from: https://github.com/tflearn/tflearn/blob/master/tflearn/datasets/cifar10.py
import cifar100
dataset = 'CIFAR100' #set book-keeping parameter

train_tup, test_tup = cifar100.load_data()

X_train, y_train = train_tup
X_validation = X_train[49000:,:]
y_validation = y_train[49000:]
X_train = X_train[:49000,:]
y_train = y_train[:49000]

X_test, y_test = test_tup

# Set training parameters
EPOCHS = 50
BATCH_SIZE = 200
NUM_TRAIN = X_train.shape[0] # set number of training examples
NUM_VAL = X_validation.shape[0] 
NUM_TEST = X_test.shape[0] 
print('Number of training examples: {}'.format(NUM_TRAIN))
print('Number of validation examples: {}'.format(NUM_VAL))
print('Number of testing examples: {}'.format(NUM_TEST))

# Loop over sizes and types to generate data

for networkSize in networkSizes:
    for networkType in networkTypes:
        runmodel((X_train,y_train,NUM_TRAIN), (X_validation,y_validation,NUM_VAL), (X_test,y_test,NUM_TEST), dataset, 
                 BATCH_SIZE, EPOCHS, num_classes = 100, networkSize = networkSize, networkType = networkType, alpha = 5e-4)

# Initialize SVHN Data
# Used code from: https://github.com/tflearn/tflearn/blob/master/tflearn/datasets/cifar10.py
import svhn
dataset = 'SVHN' #set book-keeping parameter

train_tup, test_tup = svhn.load_data()

X_train, y_train = train_tup
X_validation = X_train[72257:]
y_validation = y_train[72257:]
X_train = X_train[:60000]
y_train = y_train[:60000]

X_test, y_test = test_tup

# Set training parameters
EPOCHS = 100
BATCH_SIZE = 128
NUM_TRAIN = X_train.shape[0] # set number of training examples
NUM_VAL = X_validation.shape[0] 
NUM_TEST = X_test.shape[0] 
print('Number of training examples: {}'.format(NUM_TRAIN))
print('Number of validation examples: {}'.format(NUM_VAL))
print('Number of testing examples: {}'.format(NUM_TEST))

# Loop over sizes and types to generate data

for networkSize in networkSizes:
    for networkType in networkTypes:
        runmodel((X_train,y_train,NUM_TRAIN), (X_validation,y_validation,NUM_VAL), (X_test,y_test,NUM_TEST), dataset, 
                 BATCH_SIZE, EPOCHS, num_classes = 10, networkSize = networkSize, networkType = networkType, alpha = 1e-1)



