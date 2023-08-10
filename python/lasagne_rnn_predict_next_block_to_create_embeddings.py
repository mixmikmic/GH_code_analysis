# Python libraries
import numpy as np
import theano
import theano.tensor as Tensor
import lasagne
import random
import sys
import csv
import time
import matplotlib.pyplot as plt
import pickle
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle
# allows plots to show inline in ipython notebook
get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Import our own modules
import utils
import model_predict_block as model
import visualize
from constants import *

# hyperparameters
hidden_size = 128 # size of hidden layer of neurons
learning_rate = 1e-2
lr_decay = 0.995
reg_strength = 2e-2
grad_clip = 10
batchsize = 32
num_epochs = 8
dropout_p = 0.5
num_lstm_layers = 1
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'

HOC_NUM = 7
DATA_SZ = -1
train_data, val_data, test_data, all_data, num_timesteps, num_blocks  =utils.load_dataset_predict_block(hoc_num=HOC_NUM, data_sz=DATA_SZ)

print 'num_timesteps {}'.format(num_timesteps)

X_train, mask_train, y_train = train_data
X_val, mask_val, y_val = val_data
X_test, mask_test, y_test = test_data
print 'X_train shape {}'.format(X_train.shape)
print 'mask_train shape {}'.format(mask_train.shape)
print 'y_train shape {}'.format(y_train.shape)
print 'X_val shape {}'.format(X_val.shape)
print 'X_test shape {}'.format(X_test.shape)
print X_train[:10,:15:]
print mask_train[:10,:15]
print y_train[:10]

print utils.convert_to_block_strings(y_train[:10])

# create model
print num_blocks
train_loss_acc, compute_loss_acc, probs, generate_hidden_reps, compute_pred = model.create_model(num_timesteps, num_blocks, hidden_size, learning_rate,                                                              grad_clip, dropout_p, num_lstm_layers)

# Training!
train_losses, train_accs, train_corrected_accs, val_losses, val_accs, val_corrected_accs = model.train(train_data, val_data, train_loss_acc, compute_loss_acc,                                                              compute_pred, num_epochs=num_epochs, batchsize=batchsize, record_per_iter=True)

# Evaluate on test set
test_loss, test_raw_acc, test_corrected_acc, pred_test = model.check_accuracy(test_data, compute_loss_acc, dataset_name='test')

X_all, mask_all, _ = all_data
ast_embeddings = generate_hidden_reps(X_all, mask_all)

print ast_embeddings.shape
print X_all.shape

print ast_embeddings[:10, :10]

utils.save_ast_embeddings(ast_embeddings, HOC_NUM)

ast_embeddings_ast_row_to_ast_id_map = pickle.load('map_row_index_to_ast_id_2.pickle')
traj_mat_ast_row_to_ast_id_map = pickle.load()

# Visualize the loss and the accuracies for both training and validation sets for each epoch
num_train = train_data[0].shape[0]
visualize.plot_loss_acc('hoc' + str(HOC_NUM) + '_train', train_losses, train_corrected_accs, val_corrected_accs, learning_rate, reg_strength, num_epochs, num_train, xlabel='iterations')

# prepare data for all HOCs
X_all_hocs_mat, mask_all_hocs_mat, y_all_hocs_mat, split_indices = utils.load_dataset_predict_block_all_hocs()
# Shuffle
X_all_hocs_mat, mask_all_hocs_mat, y_all_hocs_mat = shuffle(X_all_hocs_mat, mask_all_hocs_mat, y_all_hocs_mat, random_state=0)

print X_all_hocs_mat.shape
print mask_all_hocs_mat.shape
print y_all_hocs_mat.shape
num_samples_total, num_timesteps, num_blocks = X_all_hocs_mat.shape

# trying out sklearn kfold

kf = KFold(num_samples_total, n_folds=6)
print(kf)  

# hyperparameters
hidden_size = 128 # size of hidden layer of neurons
learning_rate = 1e-2
lr_decay = 0.995
reg_strength = 2e-2
grad_clip = 10
batchsize = 8
num_epochs = 1
dropout_p = 0.5
num_lstm_layers = 1
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'

train_loss_acc, compute_loss_acc, probs, generate_hidden_reps, compute_pred = model.create_model(num_timesteps, num_blocks, hidden_size, learning_rate,                                                              grad_clip, dropout_p, num_lstm_layers)

# Training!
for train_index, val_index in kf:
    print("TRAIN:", train_index, "TEST:", val_index)
    train_data = (X_all_hocs_mat[train_index], mask_all_hocs_mat[train_index],y_all_hocs_mat[train_index])
    val_data = (X_all_hocs_mat[val_index], mask_all_hocs_mat[val_index],y_all_hocs_mat[val_index])
    
    train_losses, train_accs, train_corrected_accs, val_losses, val_accs, val_corrected_accs = model.train(train_data, val_data, train_loss_acc, compute_loss_acc,                                                                  compute_pred, num_epochs=num_epochs, batchsize=batchsize, record_per_iter=True)

X_all_hocs_mat, mask_all_hocs_mat, y_all_hocs_mat, split_indices = utils.load_dataset_predict_block_all_hocs()

ast_embeddings_all_hocs = generate_hidden_reps(X_all_hocs_mat, mask_all_hocs_mat)

utils.save_ast_embeddings_for_all_hocs(ast_embeddings_all_hocs, split_indices)

print ast_embeddings_all_hocs.shape

print ast_embeddings_all_hocs[:10, :10]

# Optionally, you could now dump the network weights to a file like this:
np.savez('../saved_models/embedding_model_for_all_hocs.npz', lasagne.layers.get_all_param_values(l_out))

# And load them again later on like this:

# with np.load('../saved_models/embedding_model_for_all_hocs.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
#     lasagne.layers.set_all_param_values(l_out, param_values)

# Create embeddings for individual hocs

# hyperparameters
hidden_size = 128 # size of hidden layer of neurons
learning_rate = 1e-2
lr_decay = 0.995
reg_strength = 2e-2
grad_clip = 10
batchsize = 32
num_epochs = 4
dropout_p = 0.5
num_lstm_layers = 1
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'

def create_embedding(hoc_num, use_forward_and_backward_lstm=False, description='indiv'):
    train_data, val_data, test_data, all_data, num_timesteps, num_blocks  =    utils.load_dataset_predict_block(hoc_num)

    X_train, mask_train, y_train = train_data
    X_val, mask_val, y_val = val_data
    X_test, mask_test, y_test = test_data
    
    train_loss_acc, compute_loss_acc, probs, generate_hidden_reps, compute_pred,l_out = model.create_model(num_timesteps, num_blocks, hidden_size, learning_rate,                                                              grad_clip, dropout_p, num_lstm_layers, use_forward_and_backward_lstm=use_forward_and_backward_lstm)
    train_losses, train_accs, train_corrected_accs, val_losses, val_accs, val_corrected_accs = model.train(train_data, val_data, train_loss_acc, compute_loss_acc,                                                              compute_pred, num_epochs=num_epochs, batchsize=batchsize, record_per_iter=True)
    test_loss, test_raw_acc, test_corrected_acc, pred_test = model.check_accuracy(test_data, compute_loss_acc, dataset_name='test')
    
    # Visualize the loss and the accuracies for both training and validation sets for each epoch
    num_train = train_data[0].shape[0]
    visualize.plot_loss_acc('hoc' + str(HOC_NUM) + '_train', train_losses, train_corrected_accs, val_corrected_accs, learning_rate, reg_strength, num_epochs, num_train, xlabel='iterations')
    
    X_all, mask_all, _ = all_data
    ast_embeddings = generate_hidden_reps(X_all, mask_all)
    utils.save_ast_embeddings(ast_embeddings, hoc_num, description=description)
    np.savez('../saved_models/indiv_embedding_model_' + str(hoc_num) + '.npz', lasagne.layers.get_all_param_values(l_out))

for hoc_num in xrange(HOC_MIN, HOC_MAX + 1):
    create_embedding(hoc_num, use_forward_and_backward_lstm=True)

for hoc_num in xrange(HOC_MIN, HOC_MAX + 1):
    create_embedding(hoc_num, use_forward_and_backward_lstm=False, description='only_forward')



