import theano.sandbox.cuda
import time
theano.sandbox.cuda.use("gpu1"), time.asctime()

import skimage
skimage.__version__ # We need at least version 0.11.3

import gzip
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import time
get_ipython().magic('matplotlib inline')

start = time.time()
npzfile = np.load('HCS_72x72.npz')
start = time.time()
cell_rows = npzfile['arr_0']
X = npzfile['arr_1']
Y = npzfile['arr_2']
print ("Loaded data in " + str(time.time() - start))
np.shape(cell_rows), np.shape(Y), np.shape(X), type(X)

Xmean = X.mean(axis = 0)
XStd = np.sqrt(X.var(axis=0))
X = (X-Xmean)/(XStd + 0.01)
Y = np.asarray(Y,dtype='int32')

idx_DMSO = np.asarray(np.recfromtxt ('DMSO_data.csv'))

np.sum(idx_DMSO)

### No DMSO

Y = Y - 1

idx_test = np.asarray(np.recfromtxt ('test_set_data.csv'))
idx_test
X_train = X[(idx_test == False) & (idx_DMSO == False)]
X_test = X[idx_test & (idx_DMSO == False)]
Y_train = Y[(idx_test == False) & (idx_DMSO == False)]
Y_test = Y[idx_test & (idx_DMSO == False)]
X_test.shape, Y_test.shape, X_train.shape, Y_train.shape #1964 and 10203

PIXELS = 72
conv = (3,3)
stride = (1,1)
pool = (2,2)

num1 = 32
num2 = 64
num3 = 128
p_drop = 0.3

CLASSES = 3

from lasagne import layers
from lasagne import nonlinearities
import theano
import theano.tensor as T
import lasagne

input_var = T.tensor4('inputs') #This is a variable needed 
l_in = lasagne.layers.InputLayer(shape=(None, 5, PIXELS, PIXELS), input_var=input_var) #None depend on batch size


conv11 = layers.Conv2DLayer(l_in, num_filters=num1, filter_size=conv)
conv11 = layers.Conv2DLayer(conv11, num_filters=num1, filter_size=conv)
pool1 = layers.MaxPool2DLayer(conv11, pool_size=pool)

conv21 = layers.Conv2DLayer(pool1, num_filters=num2, filter_size=conv)
conv22 = layers.Conv2DLayer(conv21, num_filters=num2, filter_size=conv)
pool2 = layers.MaxPool2DLayer(conv22, pool_size=pool)

conv31 = layers.Conv2DLayer(pool2, num_filters=num3, filter_size=conv)
conv32 = layers.Conv2DLayer(conv31, num_filters=num3, filter_size=conv)
pool3 = layers.MaxPool2DLayer(conv32, pool_size=pool)

hidden1 = layers.DenseLayer(layers.dropout(pool3, p_drop), num_units=200)
hidden2 = layers.DenseLayer(layers.dropout(hidden1, p_drop), num_units=200)
hidden3 = layers.DenseLayer(layers.dropout(hidden2, p_drop), num_units=50)

network = layers.DenseLayer(hidden3, num_units=CLASSES, nonlinearity=lasagne.nonlinearities.softmax)

target_var = T.ivector('targets') #The classes 0..9
prediction = layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
#print("Number of Parameters in network: {}".format(len(params)))

train_fn = theano.function([input_var, target_var], loss, updates=updates)
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import numpy as np
from skimage import transform as tf

#rots = np.deg2rad(np.asarray((90,180,0,5,-5,10,-10)))
rots = np.deg2rad(range(0,359))

dists = (-5,5)

def manipulateTrainingData(Xb):
    retX = np.zeros((Xb.shape[0], Xb.shape[1], Xb.shape[2], Xb.shape[3]), dtype='float32')
    for i in range(len(Xb)):
        rot = rots[np.random.randint(0, len(rots))]

        tf_rotate = tf.SimilarityTransform(rotation=rot)
        shift_y, shift_x = np.array((X.shape[2], X.shape[3])) / 2.
        tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])
        tform_rot = (tf_shift + (tf_rotate + tf_shift_inv))

        ## TODO add the transformations
        scale = np.random.uniform(0.9,1.10)
        d = tf.SimilarityTransform(scale=scale, translation=(np.random.randint(5),np.random.randint(5)))
        tform_other = (tform_rot + d)

        for c in range(np.shape(X)[1]):
            maxAbs = 256.0;np.max(np.abs(Xb[i,c,:,:]))
            # Needs at lease 0.11.3
            retX[i,c,:,:] = tf.warp(Xb[i,c,:,:], tform_other, preserve_range = True) # "Float Images" are only allowed to have values between -1 and 1
    return retX

############################## Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

import pandas as pd
perf = pd.DataFrame(columns=['train_loss','valid_loss','valid_accuracy', 'time'])
perf_test = pd.DataFrame(columns=['epoch','valid_acc_mean','valid_acc_std'])

np.random.seed(seed=42)
perm1 = np.random.permutation(len(Y_train))
N_split = int(len(Y_train) * 0.8)
idx_train1  = perm1[:N_split]
idx_val  = perm1[N_split:]

X_train1 = X_train[idx_train1]
y_train1 = Y_train[idx_train1]
X_val = X_train[idx_val]
y_val = Y_train[idx_val]
np.shape(X_train1), np.shape(y_train1), np.shape(X_val), np.shape(y_val)

pred_func = theano.function([input_var],[test_prediction])

import cPickle as pickle
# We iterate over epochs:
num_epochs = 510
print('Starting')
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    print('Starting')
    for batch in iterate_minibatches(X_train1, y_train1, 100, shuffle=True):
        inputs, targets = batch
        #print('Manipulating inputs '.format(np.shape(inputs)))
        dd = manipulateTrainingData(inputs)
        train_err += train_fn(dd, targets)
        train_batches += 1
    
    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 50, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
    
    time_taken = time.time() - start_time
    perf.loc[epoch] = [train_err / train_batches, val_err / train_batches, val_acc / val_batches, time_taken]
    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time_taken))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / train_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
    
    perf.to_csv('/home/dueo/Dropbox/Server_Sync/current_training.csv')
    
    ## Testing on the testset
    avg = []
    for batch in iterate_minibatches(X_test, Y_test, 100, shuffle=True):
        inputs, targets = batch
        res = pred_func(inputs)
        avg.append(np.mean(np.argmax(res[0],axis=1) == targets))
        perf_test.loc[epoch] = [epoch, np.mean(avg), np.std(avg)]
    perf_test.to_csv('/home/dueo/Dropbox/Server_Sync/current_test.csv')
    
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
    if (epoch % 10 == 0):
        np.savez('net_PAPER_aug_epoch{}_72x72large_net.pickle'.format(epoch), lasagne.layers.get_all_param_values(network))

