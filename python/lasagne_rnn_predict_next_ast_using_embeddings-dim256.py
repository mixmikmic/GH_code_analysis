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
from pprint import pprint
# allows plots to show inline in ipython notebook
get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Import our own modules
import utils
import model_predict_ast as model
import visualize
from constants import *

# hyperparameters
hidden_size = 256 # size of hidden layer of neurons
learning_rate = 3e-3
lr_decay = 0.995
reg_strength = 1e-2
grad_clip = 10
batchsize = 8
num_epochs = 1
dropout_p = 0.5
num_lstm_layers = 1
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'

def full_train_test_run(hoc_num):
    X, y, ast_maps, num_asts = utils.load_dataset_predict_ast(hoc_num, use_embeddings=True)
    num_traj, num_timesteps, embed_dim = X.shape
    # make train, val, test split
    train_data, val_data, test_data = utils.get_train_val_test_split((X,y))
    # create model
    train_loss_acc, compute_loss_acc, probs, l_out = model.create_model(num_timesteps, num_asts,                hidden_size, learning_rate, embed_dim, grad_clip, dropout_p, num_lstm_layers)
    # Training!
    train_losses, train_accs, train_corrected_accs, val_losses, val_accs, val_corrected_accs =         model.train(train_data, val_data, train_loss_acc, compute_loss_acc, num_epochs=num_epochs, batchsize=batchsize, record_per_iter=True)
    # Evaluate on test set
    test_loss, test_raw_acc, test_corrected_acc, test_corrected_acc_per_timestep, pred_test =         model.check_accuracy(test_data, compute_loss_acc, dataset_name='test', compute_acc_per_timestep_bool=True)
 
    num_train = train_data[0].shape[0]
    visualize.plot_loss_acc('hoc'+ str(hoc_num) + '_train', train_losses, train_corrected_accs, val_corrected_accs,                             learning_rate, reg_strength, num_epochs, num_train, xlabel='iterations')  
    np.savez('../saved_models/predict_ast_with_embed_model_' + str(hoc_num) + '.npz', lasagne.layers.get_all_param_values(l_out))
    return train_corrected_accs[-1], val_corrected_accs[-1], test_corrected_acc, test_corrected_acc_per_timestep

last_train_corrected_acc = {}
last_val_corrected_acc = {}
test_corrected_acc = {}
test_corrected_acc_per_timestep = {}

for hoc_num in xrange(HOC_MIN, HOC_MAX):
    print ('Training and testing predict_next_ast rnn on Hour of Code problem #{}'.format(hoc_num))
    last_train_corrected_acc[hoc_num], last_val_corrected_acc[hoc_num], test_corrected_acc[hoc_num], test_corrected_acc_per_timestep[hoc_num] = full_train_test_run(hoc_num) 
    print ('='*100)

for hoc_num in xrange(HOC_MIN + 1, HOC_MAX):
    print ('Accuracies for HOC {}:'.format(hoc_num))
    print ('last train acc: {:.2f} %'.format(last_train_corrected_acc[hoc_num] * 100))
    print ('last val acc: {:.2f} %'.format(last_val_corrected_acc[hoc_num] * 100))
    print ('overall test acc: {:.2f} %'.format(test_corrected_acc[hoc_num] * 100))
    print ('test accs time series: {}'.format(test_corrected_acc_per_timestep[hoc_num]))



