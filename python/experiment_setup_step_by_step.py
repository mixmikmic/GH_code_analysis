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

n_students = 10000
seqlen = 100
concept_tree = cdg.ConceptDependencyGraph()
concept_tree.init_default_tree(n=N_CONCEPTS)
print ("Initializing synthetic data sets...")
for policy in ['random', 'expert', 'modulo']:
    filename = "{}stud_{}seq_{}.pickle".format(n_students, seqlen, policy)
    dgen.generate_data(concept_tree, n_students=n_students, seqlen=seqlen, policy=policy, filename="{}{}".format(SYN_DATA_DIR, filename))
print ("Data generation completed. ")

dataset_name = "10000stud_100seq_modulo"

# load generated data from picke files and convert it to a format so we can feed it into an RNN for training
# NOTE: This step is not necessary if you already have the data saved in the format for RNNs. 
data = dataset_utils.load_data(filename="../synthetic_data/{}.pickle".format(dataset_name))
input_data_, output_mask_, target_data_ = dataset_utils.preprocess_data_for_rnn(data)

# Save numpy matrices to files so data loading is faster, since we don't have to do the conversion again.
dataset_utils.save_rnn_data(input_data_, output_mask_, target_data_, dataset_name)

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

# Load model with parameters initialized randomly
dmodel = dm.DynamicsModel(model_id=model_id, timesteps=100, load_checkpoint=False)

# train model for two epochs (saves checkpoint after each epoch) 
# (checkpoint saves the weights, so we can load in pretrained models.)
dmodel.train(train_data, n_epoch=2)

# Load model from latest checkpoint 
dmodel = dm.DynamicsModel(model_id=model_id, timesteps=100, load_checkpoint=True)

# train for 2 more epochs
dmodel.train(train_data, n_epoch=2)

dmodel.train(train_data, n_epoch=16)

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



