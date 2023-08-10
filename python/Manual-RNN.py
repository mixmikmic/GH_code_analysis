# Import Dependencies
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Create Constants

# Number of Inputs for each Layer
num_inputs = 2

# Number of Neurons in Hidden Layer
num_neurons = 3

# Placeholders
# Placeholder for Input at each timestamp: t0, t1, t2
x0 = tf.placeholder(tf.float32, shape=[None, num_inputs])

x1 = tf.placeholder(tf.float32, shape=[None, num_inputs])

# Weight Variable for each Timestamp
# All the weights are initialized to this.

# Input to Hidden Layer Weights
Wx = tf.Variable(tf.random_normal(shape=[num_inputs, num_neurons]))

# Hidden Layer to Output Layer Weights
# Hidden Layer at t0 to Hidden Layer at t1 weights
Wy = tf.Variable(tf.random_normal(shape=[num_neurons, num_neurons]))

# Bias
b = tf.Variable(tf.zeros([1,num_neurons]))

# Graphs
# Output at t0
y0 = tf.tanh(tf.add(tf.matmul(x0,Wx),b))

# Output at t1
y1 = tf.tanh(tf.add(tf.matmul(y0,Wy), tf.add(tf.matmul(x1,Wx),b)))

# Create Data
# Data Input at t0
x0_batch = np.array([[0,1], [2,3], [4,5]])

# Data Input at t1
x1_batch = np.array([[100,101], [102,103], [104,105]])

init = tf.global_variables_initializer()

# Run Session
with tf.Session() as sess:
    
    sess.run(init)
    
    y0_out, y1_out = sess.run([y0,y1], feed_dict={x0:x0_batch, x1:x1_batch})

y0_out

y1_out

