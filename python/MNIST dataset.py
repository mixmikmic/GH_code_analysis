# Necessary imports
import time
from IPython import display

# Numpy.
# Matplotlib for plotting images.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, figure
from PIL import Image, ImageOps
import tensorflow as tf

get_ipython().magic('matplotlib inline')

from tensorflow.examples.tutorials.mnist import input_data

# Read the mnist dataset.
mnist = input_data.read_data_sets("data/", one_hot=True)

# NOTE: MNIST is of type: tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet
# Tensorflow porvides an interface to build dataset and create batches.
print(type(mnist[0]), dir(mnist[0]))
print(type(mnist[1]), dir(mnist[1]))

# Inspect the dataset.
# Splits: Training, Validation, Testing (90/10).
image_h = 28
image_w = 28
print("Image Size: {}".format(image_h * image_w))
print("---"*11)

# Dataset size.
print("Training data Size: {}".format(mnist[0].num_examples))
print("Training Image Size: {}".format(mnist[0].images.shape))
print("---"*11)
print("Validation Size: {}".format(mnist[1].num_examples))
print("Validation Size: {}".format(mnist[1].images.shape))
print("---"*11)
print("Test Size: {}".format(mnist[2].num_examples))
print("Test Size: {}".format(mnist[2].images.shape))
print("---"*11)

# Figure.
fig = figure()

# An example of an image.
idx = 350 # Random
img_1 = np.reshape(mnist[0].images[idx], (28, 28))
fig.add_subplot(1, 3, 1)
imshow(img_1, cmap="Greys")

img_2 = np.reshape(mnist[0].images[idx+1], (28, 28))
fig.add_subplot(1, 3, 2)
imshow(img_2, cmap="Greys")

img_3 = np.reshape(mnist[0].images[idx+2], (28, 28))
fig.add_subplot(1, 3, 3)
imshow(img_3, cmap="Greys")
print("Label 9: {}".format(mnist[0].labels[idx]))
print("Label 1: {}".format(mnist[0].labels[idx+1]))
print("Label 7: {}".format(mnist[0].labels[idx+2]))

# Test a batch.
batch_size = 25
batch_X, batch_Y = mnist.train.next_batch(batch_size)
print(batch_X.shape, batch_Y.shape)

# Import encoder.
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])

print("Number of Encoded Values: {}".format(enc.n_values_))
print("Label 9: {}".format(enc.transform([[9]]).toarray()))
print("Label 1: {}".format(enc.transform([[1]]).toarray()))
print("Label 7: {}".format(enc.transform([[7]]).toarray()))

