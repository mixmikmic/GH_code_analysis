from __future__ import print_function
import numpy as np
import sys, os

sys.path.append('../')
from src import helpers
from src import digit_batches as d
from models import perceptron

train_filename = helpers.maybe_download(
    url='http://deeplearning.net/data/mnist/',
    data_root='../data/',
    filename='mnist.pkl.gz',
    expected_bytes=16168813)

import matplotlib.pyplot as plt
def plot(X):# plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(X[0].reshape(28,28), cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(X[1].reshape(28,28), cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(X[2].reshape(28,28), cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(X[3].reshape(28,28), cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()

digits = d.Digits_batches([[1],[1,2,3],[3]], batch_size=128)

X,y = digits.batches[0].next()

plot(X)

X,y = digits.batches[1].next()
plot(X)

X,y = digits.batches[2].next()
plot(X)
y

