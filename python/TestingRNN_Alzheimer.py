import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
import time
import pandas as pd
import numpy as np

N_LSTM = 64   #Number LSTM units  
N_BATCH = 100  #Batchsize
num_epochs=100  #Number of epchos 
num_inputs  = 4 #Number of features / alphabet of the input sequence

GRAD_CLIP = 100 #Clipping gradient in backpropagation
LEARNING_RATE = 0.01 #Learning rate

df = pd.read_csv('Zeitreihen_id_mani.csv', sep =" ") 
df[0:4], df.shape

# X.shape 1759
idx_train = 700
idx_val   = 700 + 350

X_ = np.asarray(df.ix[:,'X2':'X37']) + 2 #+2 so that all goes from 0 to 3
y_ = df.ix[:,'playerId']
X_.shape, y_.shape
LENGTH = X_.shape[1] #Length of Sequence

np.histogram(X_, bins=(0,1,2,3,4,5)),set(np.reshape(X_,-1))

y_set = list(set(y_))
num_to_ix = { ch:i for i,ch in enumerate(y_set) }
ix_to_num = { i:ch for i,ch in enumerate(y_set) }
y_d = np.asarray([num_to_ix[y_[i]] for i in range(len(y_))],dtype='int32')
num_classes  = len(ix_to_num) 
np.histogram(y_d, bins=range(30))

pd.crosstab(y_,1)

idx = np.random.permutation(X_.shape[0])
y = y_d[idx]
X = X_[idx]

X.shape

def makeCategorical(X_in):
    X_out = np.zeros((X_in.shape[0], X_in.shape[1], num_inputs),dtype='float32')
    for i in range(X_in.shape[0]):
        for j in range(X_in.shape[1]):
            X_out[i,j,X_in[i,j]] = 1
    return X_out

#X_ = np.asarray(np.reshape(X,(X.shape[0],X.shape[1],1)), dtype='float32') #Not categorical

#X_train = X_[0:idx_train,:]
X_train = makeCategorical(X[0:idx_train,:])
y_train = y[0:idx_train]

#X_val = X_[idx_train:idx_val]
X_val = makeCategorical(X[idx_train:idx_val])
y_val = y[idx_train:idx_val]

#X_val = X_[idx_train:idx_val]
X_test = makeCategorical(X[idx_val:])
y_test = y[idx_val:]

X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

#We leave the batch size open in the definition of the network
l_in = InputLayer(shape=(None, LENGTH, num_inputs)) #Example: (10, 36, 1)
current_bs = l_in.input_var.shape[0]

l_lstm = LSTMLayer(l_in, N_LSTM, grad_clipping=GRAD_CLIP, nonlinearity=lasagne.nonlinearities.tanh) 
#Shape (Batches, LENGTH, N_LSTM) Example: (10, 36, 12)

#l_lstm_1 =  LSTMLayer(l_lstm, N_LSTM, grad_clipping=GRAD_CLIP, nonlinearity=lasagne.nonlinearities.tanh)

# See https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
# The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
l_shp = lasagne.layers.SliceLayer(l_lstm, -1, 1)
# Shape (10,12)

l_out = DenseLayer(l_shp, num_units=num_classes, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax) 
#Shape (10, 21)
#l_out = ReshapeLayer(l_dense, (current_bs, LENGTH, num_classes)) #(10, 36, 21)

input = l_in.input_var
print("Defined network ...")

print("Shapes\n-------\ninput: {}\nLSTM in{} out {}\nSlice Layer in{} out{}\nDense Layer in{} out{}".format(
l_in.shape,
l_lstm.input_shapes,
l_lstm.output_shape,
l_shp.input_shape,
l_shp.output_shape,
l_out.input_shape,
l_out.output_shape))

# Testing the network (are the number of layers like what one expects)
preds = theano.function([l_in.input_var], lasagne.layers.get_output(l_out))
Xd = np.asarray(makeCategorical(X[0:10,:]))
res = preds(Xd)
res.shape, res[0], np.sum(res[0])

# lasagne.layers.get_output produces a variable for the output of the net
target_values = T.ivector('target_output')
network_output = lasagne.layers.get_output(l_out)
cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()
cost_function = theano.function([l_in.input_var, target_values], cost)

#For testing only 
#we have an untrained network, so the costs should be like random
random_loss = cost_function(makeCategorical(X),y)
random_loss

all_params = lasagne.layers.get_all_params(l_out)
updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)

test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_values)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_values), dtype=theano.config.floatX)
val_fn = theano.function([l_in.input_var, target_values], [test_loss, test_acc])

test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_values))
val_fn = theano.function([l_in.input_var, target_values], [cost, test_acc]) 

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
perf = pd.DataFrame(columns=['train_loss','valid_loss','valid_accuracy'])
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 100, shuffle=True):
        inputs, targets = batch
        train_err += train(inputs, targets)
        train_batches += 1
        
    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 250, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
    
    perf.loc[epoch] = [train_err / train_batches, val_err / val_batches, val_acc / val_batches]
    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

get_ipython().magic('matplotlib inline')
perf[['train_loss','valid_loss','valid_accuracy']].plot(title='Performance during Training')
perf[['valid_accuracy']].plot(title='Performance during Training')

p = preds(X_val) 
np.sum(np.argmax(p,axis=1) == y_val)*1.0/len(y_val)

p = preds(X_test)
np.sum(np.argmax(p,axis=1) == y_test)*1.0/len(y_test)

from sklearn.metrics import confusion_matrix
m = confusion_matrix(np.argmax(p,axis=1), y_test)
df = pd.DataFrame(m)
cm_normalized = m.astype('float') / ((m.sum(axis=0)[np.newaxis,:])[0])

diag = [cm_normalized[i,i] for i in range(21)]

np.mean(diag), 1.0/21.0

