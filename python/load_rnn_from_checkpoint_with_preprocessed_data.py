import sys
print sys.executable
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('reload_ext autoreload')

import tensorflow as tf
import tflearn
import numpy as np
import time

import concept_dependency_graph as cdg
import dataset_utils
import dynamics_model_class as dm
import data_generator as dgen
from filepaths import *
from constants import *

dataset_name = "10000stud_100seq_modulo"

# load the numpy matrices
input_data_, output_mask_, target_data_ = dataset_utils.load_rnn_data(dataset_name)

print input_data_.shape
print target_data_.shape

from sklearn.model_selection import train_test_split

x_train, x_test, mask_train, mask_test, y_train, y_test = train_test_split(input_data_, output_mask_, target_data_, test_size=0.1, random_state=42)

train_data = (x_train, mask_train, y_train)

import models_dict_utils

# Each RNN model can be identified by its model_id string. 
# We will save checkpoints separately for each model. 
# Models can have different architectures, parameter dimensions etc. and are specified in models_dict.json
model_id = "learned_from_modulo"

# Specify input / output dimensions and hidden size
n_timesteps = 100
n_inputdim = 20
n_outputdim = 10
n_hidden = 32

# If you are creating a new RNN model or just to check if it already exists:
# Only needs to be done once for each model

models_dict_utils.check_model_exists_or_create_new(model_id, n_inputdim, n_hidden, n_outputdim)

# Load model from latest checkpoint 
dmodel = dm.DynamicsModel(model_id=model_id, timesteps=100, load_checkpoint=True)

# important to cast preds as numpy array. 
preds = np.array(dmodel.predict(x_test[:1]))

print preds.shape

print preds[0,0]

# Load model with different number of timesteps from checkpoint
# Since RNN weights don't depend on # timesteps (weights are the same across time), we can load in the weights for 
# any number of timesteps. The timesteps parameter describes the # of timesteps in the input data.
generator_model = dm.DynamicsModel(model_id=model_id, timesteps=1, load_checkpoint=True)

# make a prediction
preds = generator_model.predict(x_test[:1,:1, :])

print preds[0][0]



