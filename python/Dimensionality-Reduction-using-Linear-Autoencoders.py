import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
get_ipython().magic('matplotlib inline')

# Make Random Dataset
# 2 Classes
# 3 features
# Num Samples = 100
data = make_blobs(n_samples=100, n_features=3, centers=2, random_state=101)

type(data)

data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[0])

scaled_data

# Features
X1 = scaled_data[:,0]
X2 = scaled_data[:,1]
X3 = scaled_data[:,2]

from mpl_toolkits.mplot3d import Axes3D

# Plot 3-D Data
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X1, X2, X3, c=data[1])

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# Neurons in Input Layer
n_inputs = 3

# Neurons in Hidden Layer
hidden = 2

# Neurons in Output Layer
n_outputs = 3

# Learning Rate
lr = 0.01

# Input Data
X = tf.placeholder(tf.float32, shape=[None,n_inputs])

# Hidden Layer
# Aim is to take 3-D data andconvert to 2-D by learning representations
h1 = fully_connected(X, hidden, activation_fn=None)

# Output is rescaled to 3-D from 2-D
output = fully_connected(h1, n_outputs, activation_fn=None)

# Loss Function
# This checks that how far the Ouptut is from reproducing the Input X
loss = tf.reduce_mean(tf.square(output - X))

optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

init = tf.global_variables_initializer()

n_steps = 5000

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(n_steps):
        _ , loss_val = sess.run([optimizer,loss], feed_dict={X: scaled_data})
        
        if step % 100 == 0:
            print('STEP: {0}\t , LOSS: {1}'.format(step, loss_val))
    
    # Use Hidden Layer to get the 2-D Data
    output_2d = h1.eval(feed_dict={X: scaled_data})

# Plot the 2-D Data
plt.scatter(output_2d[:,0], output_2d[:,1], c=data[1])

