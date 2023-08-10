## import Matpolt with PyQt4 Backend
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Run some setup code for this notebook.
import random
import numpy as np
import pandas as pd
from mnist import MNIST

import sys
sys.path.append('./..')
from py_model.two_layer_mlp import NeuralNetwork

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

mndata = MNIST("./../data/MNIST") # Current Path
train_X, train_y = mndata.load_training()
test_X, test_y = mndata.load_testing()

train_X = np.array(train_X)
train_y = np.array(train_y)
test_X = np.array(test_X)
test_y = np.array(test_y)

print(train_X.shape, train_y.shape)
print(test_X.shape, test_y.shape)

test_y[:10]

layer1 = {'layer': (28*28, 2048), 'activation': lambda x: np.maximum(0, x) }  # ReLu
layer2 = {'layer': (2048, 10)} 
#output layer Softmax

nn = NeuralNetwork(layer1, layer2)

nn.train(train_X, train_y, learning_rate=0.0001, reg=0.001, num_iters=1000, batch_size=50, verbose=True)

# Accuracy
val_acc = (nn.predict(test_X) == test_y).mean()
print("Accuracy : " + str(val_acc*100) + "%")

# Plot Loss History
plt.plot(nn.loss_history)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()



