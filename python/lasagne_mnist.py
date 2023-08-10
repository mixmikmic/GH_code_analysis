from helper import *
import numpy as np
import theano
import theano.tensor as T
import lasagne

#Load the dataset - function defined in helper.py
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_dataset()

#Recipes
#In the end the models will be Theano expressions
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.init import GlorotUniform

def build_mlp(input_var=None):
    #shape: batchsize, channels, rows, columns
    #none: automatically deduced, as in Tensorflow
    l_in        = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    l_in_drop   = DropoutLayer(l_in, p=0.2)
    l_hid1      = DenseLayer(l_in_drop, num_units=800, nonlinearity=rectify, W=GlorotUniform())
    l_hid1_drop = DropoutLayer(l_hid1, p=0.5)
    l_hid2      = DenseLayer(l_hid1_drop, num_units=800, nonlinearity=rectify)
    l_hid2_drop = DropoutLayer(l_hid2, p=0.5)
    l_out       = DenseLayer(l_hid2_drop, num_units=10, nonlinearity=softmax)
    return l_out


def build_cnn(input_var=None):
    network = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    network = Conv2DLayer(network, num_filters=32, filter_size=(5,5), nonlinearity=rectify)
    
    network = MaxPool2DLayer(network, pool_size=(2,2))
    network = Conv2DLayer(network, num_filters=32, filter_size=(5,5), nonlinearity=rectify)
    network = MaxPool2DLayer(network, pool_size=(2,2))
    
    #normal MLP
    network = DropoutLayer(network, p=.5)
    network = DenseLayer(network, num_units=256, nonlinearity=rectify)
    network = DropoutLayer(network, p=.5)
    network = DenseLayer(network, num_units=10, nonlinearity=softmax)
    return network

#Create Theano expression with our entry-points
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
network = build_cnn(input_var) #or build_mlp

#loss function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

#calculate updates using loss function
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

#compiles train function
train_fn = theano.function([input_var, target_var], loss, updates=updates)

#test function - deterministic deactivates dropout
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var))
test_fn = theano.function([input_var, target_var], [test_loss, test_acc])

#Training loop
#iterate_minibatches is defined in helper.py
num_epochs = 100
for epoch in range(num_epochs):
    train_err = 0
    train_batches = 0
    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1
    
    train_err = train_err / train_batches
    
    #End of epoch, we show the results so far
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        inputs, targets = batch
        err, acc = test_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
    val_err = val_err / val_batches
    val_acc = val_acc / val_batches
    
    print("Epoch ", epoch, ":")
    print("Training Loss: ", train_err)
    print("Validation Loss: ", val_err)
    print("Validation Accuracy: ", val_acc)
    
        



