from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf

sys.path.append('..')
import deepomics

from deepomics import neuralnetwork as nn
from deepomics import utils, fit

fname = '../data/FreyFaces/frey_rawface.mat'

from scipy.io import loadmat
matfile = loadmat(fname)
all_data = (matfile['ff'] / 255.).T
indices = np.arange(len(all_data))
np.random.shuffle(indices)

# split dataset into train and validation 
X_train = all_data[indices[:1500]]
X_valid = all_data[indices[1500:]]

# place data as a dictionary
train = {'inputs': X_train, 'targets': X_train}
valid = {'inputs': X_valid, 'targets': X_valid}

width = 20
height = 28
input_shape = [None, width*height]
output_shape = [None, width*height]

def model(input_shape, output_shape):

    # create model
    layer1 = {'layer': 'input', #41
            'input_shape': input_shape
            }
    layer2 = {'layer': 'dense',
            'num_units': 128,
            'activation': 'relu',
            #'dropout': 0.1,
            }
    layer3 = {'layer': 'variational_normal',
            'num_units': 30,
            'name': 'Z',
            }
    layer4 = {'layer': 'dense',
            'num_units': 128,
            'activation': 'relu',
            #'dropout': 0.1,
            }
    layer5 = {'layer': 'variational_normal',
            'num_units': output_shape[1],
            'activation': 'sigmoid',
            'name': 'X'
             }
    
    #from tfomics import build_network
    model_layers = [layer1, layer2, layer3, layer4, layer5]

    # optimization parameters
    optimization = {"objective": "elbo_gaussian_gaussian",
                  "optimizer": "adam",
                  "learning_rate": 0.0003,
                  "beta1": 0.9, 
                  #"l2": 1e-6,
                  }
    return model_layers, optimization


# get model info
model_layers, optimization = model(input_shape, output_shape)

# build neural network class
nnmodel = nn.NeuralNet(seed=247)
nnmodel.build_layers(model_layers, optimization, supervised=False)
nnmodel.inspect_layers()

# compile neural trainer
model_save_path = os.path.join('../results', 'frey')
nntrainer = nn.NeuralTrainer(nnmodel, save=None, file_path=model_save_path)

# initialize session
sess = utils.initialize_session()

data = {'train': train, 'valid': valid}
fit.train_minibatch(sess, nntrainer, data, batch_size=128,
                    num_epochs=500, patience=20, verbose=1, shuffle=True)

# randomly select a set number of training samples
num_grid = 10
shuffle = np.random.permutation(X_train.shape[0])
X_data = X_train[shuffle[:num_grid*num_grid]]

# get the generated images about the latent space where the training samples mapped to
samples = nntrainer.get_activations(sess, {'inputs': X_data}, layer='X')

# plot training samples and generated trainingsamples
fig = plt.figure()
ax = plt.subplot(1,2,1);
ax.imshow((X_data.reshape(num_grid, num_grid, 28, 20)
                   .transpose(0, 2, 1, 3)
                   .reshape(num_grid*28, num_grid*20)), cmap='gray')
ax.axis('off')
ax = plt.subplot(1,2,2);
ax.imshow((samples.reshape(num_grid, num_grid, 28, 20)
                   .transpose(0, 2, 1, 3)
                   .reshape(num_grid*28, num_grid*20)), cmap='gray')
ax.axis('off')
fig.set_size_inches(15,15)

# extract latent space for training data
Z = nntrainer.get_activations(sess, {'inputs': X_train}, layer='Z_mu')

# perform PCA on latent space
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Z_reduce = pca.fit_transform(Z)

# plot reduced latent space
fig = plt.figure()
plt.scatter(Z_reduce[:,0], Z_reduce[:,1], alpha=0.2)



