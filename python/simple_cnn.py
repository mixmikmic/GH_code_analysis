# A bit of setup

# Usual imports
import time
import numpy as np
import matplotlib.pyplot as plt

# Notebook plotting magic
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# For auto-reloading external modules
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Deep learning related
import theano
import theano.tensor as T
import lasagne

# My modules
import generate_data as d

def rel_error(x, y):
    """ Returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8. np.abs(x) + np.abs(y))))

def load_dataset(num=5):
    """
    Load a bit of data from SALAMI.
    Argument: num (number of songs to load. Default=5)
    Returns: X_train, y_train, X_val, y_val, X_test, y_test
    """
    X, y = d.get_data(num)

    # Keep last 6000 data points for test
    X_test, y_test = X[-6000:], y[-6000:]
    X_train, y_train = X[:-6000], y[:-6000]

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # Make column vectors
    y_train = y_train[:,np.newaxis]
    y_val   = y_val[:,np.newaxis]
    y_test  = y_test[:,np.newaxis]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_cnn(input_var=None):
    """
    Build the CNN architecture.
    """

    # Make an input layer
    network = lasagne.layers.InputLayer(
        shape=(
            None,
            1,
            20,
            515
            ),
        input_var=input_var
        )

    # Add a conv layer
    network = lasagne.layers.Conv2DLayer(
        network,           # Incoming
        num_filters=32,    # Number of convolution filters to use
        filter_size=(5, 5),
        stride=(1, 1),     # Stride fo (1,1)
        pad='same',        # Keep output size same as input
        nonlinearity=lasagne.nonlinearities.rectify, # ReLU
        W=lasagne.init.GlorotUniform()   # W initialization
        )

    # Apply max-pooling of factor 2 in second dimension
    network = lasagne.layers.MaxPool2DLayer(
        network, pool_size=(1, 2)
        )
    # Then a fully-connected layer of 256 units with 50% dropout on its inputs
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify
        )
    # Finally add a 1-unit softmax output layer
    network = lasagne.layers.DenseLayer(
        network,
        num_units=1,
        nonlinearity=lasagne.nonlinearities.softmax
        )

    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Generate a minibatch.
    Arguments: inputs    (numpy array)
               targets   (numpy array)
               batchsize (int)
               shuffle   (bool, default=False)   
    Returns:   inputs[excerpt], targets[excerpt]
    """
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

# Theano config
theano.config.floatX = 'float32'

# Load the dataset
print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(3)

# Print the dimensions
for datapt in [X_train, y_train, X_val, y_val, X_test, y_test]:
    print datapt.shape

# Parse dimensions
n_train  = y_train.shape[0]
n_val    = y_val.shape[0]
n_test   = y_test.shape[0]
n_chan   = X_train.shape[1]
n_feats  = X_train.shape[2]
n_frames = X_train.shape[3]

print "n_train  = {0}".format(n_train)
print "n_val    = {0}".format(n_val)
print "n_test   = {0}".format(n_test)
print "n_chan   = {0}".format(n_chan)
print "n_feats  = {0}".format(n_feats)
print "n_frames = {0}".format(n_frames)

# Prepare Theano variables for inputs and targets
input_var  = T.tensor4( name='inputs' )
target_var = T.fcol( name='targets' )

# Create neural network model (depending on first command line parameter)
print("Building model and compiling functions..."),
network = build_cnn(input_var)
print("Done.")

# Create a loss expression for training, i.e., a scalar objective we want to minimize
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.squared_error(prediction, target_var)
loss = loss.mean()

# Create update expressions for training
# Here, we'll use adam
params  = lasagne.layers.get_all_params(
    network,
    trainable=True
)
updates = lasagne.updates.adam(
    loss,
    params
)

# Create a loss expression for validation/testing.
# The crucial difference here is that we do a deterministic forward pass
# through the network, disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)

test_loss = lasagne.objectives.squared_error(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()

# As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function(
    [input_var, target_var],
    loss,
    updates=updates,
    allow_input_downcast=True
)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function(
    [input_var, target_var],
    [test_loss, test_acc],
    allow_input_downcast=True
)

num_epochs = 1

# Finally, launch the training loop.
print("Starting training...")

# We iterate over epochs:
for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    
    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))
print("Done training.")    

# After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))

# Optionally, you could now dump the network weights to a file like this:
# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
#
# And load them again later on like this:
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)

trained_params = lasagne.layers.get_all_param_values(network)

