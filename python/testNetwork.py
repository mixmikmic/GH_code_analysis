# Import necessary libraries
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Load MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")

caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
caps1_n_dims = 8

conv1 = tf.layers.conv2d(X, name="conv1", filters=256, kernel_size=9, strides=1, padding="valid", activation=tf.nn.relu)
conv2 = tf.layers.conv2d(conv1, name="conv2", filters=caps1_n_maps*caps1_n_dims, kernel_size=9, strides=2, padding="valid", activation=tf.nn.relu)

caps1_flat = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_flat")
caps1_flat



