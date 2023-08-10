import numpy as np
import scipy.io
from lasagne.layers import get_output, InputLayer, DenseLayer, GaussianNoiseLayer
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, linear
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo
from nolearn.lasagne import visualize
from nolearn.lasagne import visualize
from nolearn.lasagne.visualize import plot_loss
import theano
from lasagne.init import GlorotUniform
from plottingUtils import tile_raster_images

get_ipython().magic('pylab')

get_ipython().magic('matplotlib inline')

#download the data
import urllib.request
import os
if not os.path.exists("Olshausen.mat"):
    urllib.request.urlretrieve ("http://redwood.berkeley.edu/bruno/sparsenet/IMAGES.mat", "Olshausen.mat")

theImages = scipy.io.loadmat('Olshausen.mat')['IMAGES']

def get_random_patch(images, patchsize=12):
    """extracts a single random patch from the 3d matrix of images with shape: (x, y, samples)"""
    q,_, N = images.shape
    r = np.random.randint(N)
    x, y =  np.random.randint(q-patchsize,  size=(2,)) # q-N to stay within the image

    return images[x:x+patchsize,y:y+patchsize, r]

patchSize= 12
sampleSize = 500000
X = np.stack([get_random_patch(theImages,patchSize) for i in range(sampleSize)], axis=0)
X_out = X.reshape((X.shape[0], -1)) # flatten to compare vs last layer

"""first experiment: a linear encoder/decoder, squared loss"""
encode_size = 50

# to get tied weights in the encoder/decoder, create this shared weightMatrix
sharedWeights = theano.shared(GlorotUniform().sample(shape=(X.shape[1]**2, encode_size))) #

layers = [
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2])}),
    (DenseLayer, {'name': 'encoder', 'num_units': encode_size, 'nonlinearity': rectify, 'W': sharedWeights }),
    (DenseLayer, {'name': 'decoder','num_units': patchSize**2, 'nonlinearity': linear, 'W': sharedWeights.T}),
]

ae1 = NeuralNet(
    layers=layers,
    max_epochs=100,
    update=nesterov_momentum,
    update_learning_rate=0.05,
    update_momentum=0.975,
    regression=True,
    verbose=1
)

ae1.fit(X, X_out);

plot_loss(ae1)
plt.ylim([0.008, 0.010])

X_pred = ae1.predict(X)
tile_raster_images(X_pred[0:9,:], (patchSize,patchSize), (3,3), tile_spacing=(1,1))
title('reconstructed')
tile_raster_images(X_out[0:9,:], (patchSize,patchSize), (3,3), tile_spacing=(1,1));
title('original');

W_encode = ae1.layers_['encoder'].W.get_value()
tile_raster_images(W_encode.T, (patchSize,patchSize), (7,8), tile_spacing=(1,1)); title('encoder filters')

sigma = 0.5 #corrupt the data with gaussian noise
encode_size = 200

# again, tied weights in the en/decoder
sharedWeights2 = theano.shared(GlorotUniform().sample(shape=(X.shape[1]**2, encode_size))) #
layers2 = [
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2])}),
    (GaussianNoiseLayer, {'name': 'corrupt', 'sigma': sigma}),
    (DenseLayer, {'name': 'encoder', 'num_units': encode_size, 'nonlinearity': rectify, 'W': sharedWeights2 }),
    (DenseLayer, {'name': 'decoder','num_units': patchSize**2, 'nonlinearity': linear, 'W': sharedWeights2.T}),
]

ae2 = NeuralNet(
    layers=layers2,
    max_epochs=100,
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,
    regression=True,
    verbose=1
)

ae2.fit(X, X_out);

plot_loss(ae2)#;ylim([0.005,0.01])

# predictions
X_pred = ae2.predict(X)
tile_raster_images(X_pred[0:100,:], (patchSize,patchSize), (10,10), tile_spacing=(1,1)); title('reconstructed')
tile_raster_images(X_out[0:100,:], (patchSize,patchSize), (10,10), tile_spacing=(1,1)); title('target')

#filters
W_encode = ae2.layers_['encoder'].W.get_value()
tile_raster_images(W_encode.T, (patchSize,patchSize), (10,10), tile_spacing=(1,1)); title('filters');

