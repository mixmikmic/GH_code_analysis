# Import Python libraries
import numpy as np
import theano
import theano.tensor as Tensor
import lasagne
import time
import cPickle as pickle

# allows plots to show inline in ipython notebook
get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Import own modules
import data_utils, visualize
import lasagne_model_predict_country as cnn_model

# Model hyperparameters
cnn_architecture = "complex_cnn"
num_filters = 32
filter_width = 3 # can be integer or tuple
pool_width = 2 
stride_width = 1 # can be integer or tuple
padding = 'full'  # can be integer or tuple or 'full', 'same', 'valid'
hidden_size = 256 # size of hidden layer of neurons
dropout_p = 0.0
# lr_decay = 0.995
reg_strength = 0
# grad_clip = 10

# Optimization hyperparams
# LEARNING_RATE = 1e-2
LEARNING_RATE = 0.045

USE_OPTIMIZER = "nesterov_momentum"
# USE_OPTIMIZER = "adam"
# (1) Nesterov Momentum
MOMENTUM = 0.9
# (2) Adam
beta1=0.9
beta2=0.999
epsilon=1e-08
# Optimizer config
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'

# Training parameters
batchsize = 1000
num_epochs = 8
record_per_iter = True  # save train and val loss/accuracy after each batch runthrough

# Load Data Set

# DATA_BATCH = '000_small_'
# DATA_SIZE = '48by32'
# DATA_SET = DATA_BATCH + DATA_SIZE
# NUM_CLASSES = 5

DATA_SET = 'subset_48by32_5'
NUM_CLASSES = 5
NUM_BATCHES = 6
USE_BATCH = [0, 1, 2]

print ('Data Set:', DATA_SET)
print ('Num classes:', NUM_CLASSES)
print ('Batches: {}'.format(USE_BATCH))
print ('Preparing Data Set....')

X = None
Y = None

for batch_num in USE_BATCH:
    X_input_filename = 'data_maps/' + DATA_SET + '/x_input_' + str(batch_num) + '.npy'
    Y_output_filename = 'data_maps/' + DATA_SET + '/y_labels_' + str(batch_num) + '.npy'

    X_batch = data_utils.load_npy_file(X_input_filename)
    Y_batch = data_utils.load_npy_file(Y_output_filename)
    
    if X is None:
        X = X_batch
    else:
        X = np.vstack((X, X_batch))
        
    if Y is None:
        Y = Y_batch
    else:
        Y = np.hstack((Y, Y_batch))  # use hstack because 1-d array is represented as a row vector internally
    
    
print 'X: {}'.format(X.shape)
print 'Y: {}'.format(Y.shape)
print 'Y sample ', Y[:10]

num_samples, H, W, C = X.shape

# swap C and H axes --> expected input
X = np.swapaxes(X, 1, 3)  # (num_samples, C, W, H)
X -= np.mean(X, axis = 0)  # Data Preprocessing: mean subtraction
X /= np.std(X, axis = 0)  # Normalization

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

print ('X_train', X_train.shape)
print ('y_train', y_train.shape)
print ('X_val', X_val.shape)
print ('y_val', y_val.shape)
print ('X_test', X_test.shape)
print ('y_test', y_test.shape)

# Create model and compile train and val functions
train_fn, val_fn, l_out = cnn_model.main_create_model(C, W, H, NUM_CLASSES, cnn_architecture=cnn_architecture, num_filters=num_filters, filter_width=filter_width, pool_width=pool_width, stride=stride_width, pad=padding, hidden_size=hidden_size, dropout=dropout_p, use_optimizer=USE_OPTIMIZER, learning_rate=LEARNING_RATE, momentum=MOMENTUM, beta1=beta1, beta2=beta2, epsilon=epsilon)

# Train the model.
train_err_list, train_acc_list, val_err_list, val_acc_list, epochs_train_err_list, epochs_train_acc_list, epochs_val_err_list, epochs_val_acc_list = cnn_model.train(num_epochs, batchsize, num_train, num_val, USE_OPTIMIZER, train_fn, val_fn, X_train, y_train, X_val, y_val, record_per_iter=record_per_iter)

# After training, we compute and print the test error:
print ("Test Size: {}".format(num_test) )
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
num_train = X_train.shape[0]
if record_per_iter:
    xlabel = "iterations"
else:
    xlabel = "epochs"
# Printing training losses and training + validation accuracies
data_set_name = DATA_SET + 'batch'
for batch_num in USE_BATCH:
    data_set_name += "_"
    data_set_name += str(batch_num)
visualize.plot_loss_acc(data_set_name, train_err_list, train_acc_list, val_acc_list, LEARNING_RATE, reg_strength, num_epochs, num_train, xlabel=xlabel)

# Visualize the loss and the accuracies for both training and validation sets for each epoch
num_train = X_train.shape[0]
xlabel = "epochs"
# Printing training losses and training + validation accuracies
data_set_name = DATA_SET + 'batch_' + str(USE_BATCH) + '_on_epochs' + '_' + USE_OPTIMIZER
visualize.plot_loss_acc(data_set_name, epochs_train_err_list, epochs_train_acc_list, epochs_val_acc_list, LEARNING_RATE, reg_strength, num_epochs, num_train, xlabel=xlabel)

# Store params
model_filename = 'model_weights/' + DATA_SET + 'batch_' + str(USE_BATCH) + '_' + USE_OPTIMIZER + '.npz'
np.savez(model_filename, lasagne.layers.get_all_param_values(l_out))

# To load
# with np.load(model_filename) as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

import cPickle as pickle

pickle_filename = 'loss_acc_values/' + DATA_SET + 'batch_' + str(USE_BATCH) + '_' + USE_OPTIMIZER + '.pickle'
pickle.dump( (train_err_list, train_acc_list, val_err_list, val_acc_list), open(pickle_filename, 'wb') )



