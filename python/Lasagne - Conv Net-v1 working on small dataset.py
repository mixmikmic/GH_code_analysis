# Import Python libraries
import numpy as np
import theano
import theano.tensor as Tensor
import lasagne
import time

# Import own modules
import data_utils
import lasagne_model_predict_country as cnn_model

# Model hyperparameters
num_filters = 32
filter_width = 5
pool_width = 2
hidden_size = 256 # size of hidden layer of neurons
dropout_p = 0.0
# lr_decay = 0.995
# reg_strength = 2e-2
# grad_clip = 10

# Optimization hyperparams
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'

# Training parameters
batchsize = 32
num_epochs = 8

# Load Data Set

DATA_BATCH = '000_small_'
DATA_SIZE = '48by32'
NUM_CLASSES = 5


DATA_SET = DATA_BATCH + DATA_SIZE
print 'Data Set:', DATA_SET
print 'Num classes:', NUM_CLASSES
print 'Preparing Data Set....'

X_input_filename = 'data_maps/' + DATA_SET + '/x_input.npy'
Y_output_filename = 'data_maps/' + DATA_SET + '/y_labels.npy'

X = data_utils.load_npy_file(X_input_filename)
Y = data_utils.load_npy_file(Y_output_filename)
# print 'X: {}'.format(X.shape)
# print 'Y: {}'.format(Y.shape)
# print 'Y sample ', Y[:10]

num_samples, H, W, C = X.shape

# swap C and H axes --> expected input
X = np.swapaxes(X, 1, 3)  # (num_samples, C, W, H)
X -= np.mean(X, axis = 0)  # Data Preprocessing: mean subtraction

#Splitting into train, val, test sets

num_train = int(num_samples * 0.8)
num_val = int(num_samples * 0.1)
num_test = num_samples - num_train - num_val

# print 'num_train: %d, num_val: %d, num_test: %d' % (num_train, num_val, num_test)

X_train = X[:num_train]
X_val = X[num_train:num_train+num_val]
X_test = X[num_train+num_val:]

y_train = Y[:num_train]
y_val = Y[num_train:num_train+num_val]
y_test = Y[num_train+num_val:]

print 'X_train', X_train.shape
print 'y_train', y_train.shape
print 'X_val', X_val.shape
print 'X_test', X_test.shape

# Prepare Theano variables for inputs and targets
input_var = Tensor.tensor4('inputs')
target_var = Tensor.ivector('targets')

print('Building network...')

# Create neural network model
l_in, l_out = cnn_model.build_cnn(C, W, H, NUM_CLASSES, num_filters=num_filters, filter_width=filter_width, pool_width=pool_width, hidden_size=hidden_size, dropout=dropout_p, inputVar = input_var)

print('Compiling functions...')

# Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(l_out)
loss = Tensor.nnet.categorical_crossentropy(prediction, target_var)
# loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
acc = Tensor.mean(Tensor.eq(Tensor.argmax(prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

# Return predictions in a function
pred_fn = theano.function([l_in.input_var], prediction)

 # Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=LEARNING_RATE, momentum=MOMENTUM)


# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
# test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = Tensor.nnet.categorical_crossentropy(test_prediction, target_var)

test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_acc = Tensor.mean(Tensor.eq(Tensor.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

 # Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], [loss, acc], updates=updates)
# train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

print('Compiling Finished!')

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:

for epoch in range(num_epochs):
    # 1) In each epoch, we do a full pass over the training data:
    train_err = 0
    train_acc = 0
    train_batches = 0
    start_time = time.time()
    
    for batch in data_utils.iterate_minibatches(X_train, y_train, batchsize, shuffle=True):
        inputs, targets = batch
        err, acc = train_fn(inputs, targets)
        prediction = pred_fn(inputs)
        train_err += err
        train_acc += acc
        train_batches += 1
            
    # 2) And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in data_utils.iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
    
    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  training accuracy:\t\t{:.2f} %".format(train_acc))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))

print('Training finished!')

    
# After training, we compute and print the test error:
print('Testing...')
test_err = 0
test_acc = 0
test_batches = 0
for batch in data_utils.iterate_minibatches(X_test, y_test, batchsize, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
    
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))

# Visualize the loss and the accuracies for both training and validation sets for each epoch
num_train = train_data[0].shape[0]
visualize.plot_loss_acc('subset_5_train', train_losses, train_corrected_accs, val_corrected_accs, learning_rate, reg_strength, num_epochs, num_train, xlabel='iterations')

