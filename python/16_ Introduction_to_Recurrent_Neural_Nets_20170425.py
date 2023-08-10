import warnings
warnings.filterwarnings("ignore")
import datetime
import pandas as pd
# import pandas.io.data
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sys
import sompylib.sompy as SOM# from pandas import Series, DataFrame

from ipywidgets import interact, HTML, FloatSlider

get_ipython().magic('matplotlib inline')

# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
class RNN:
    def setup(self,hidden_size,input_size):
        Wxh = np.random.randn(hidden_size, input_size)*0.01 # input to hidden
        Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
        Why = np.random.randn(input_size, hidden_size)*0.01 # hidden to output
        self.W_hh = Whh
        self.h = np.zeros((hidden_size))
        self.W_xh = Wxh
        self.W_hy = Why
    def step(self, x):
        # update the hidden state
        self.h = np.tanh(np.dot(self.h,self.W_hh) + np.dot(self.W_xh, x)).T
#         print self.h.shape
        # compute the output vector
        y = np.dot(self.W_hy, self.h)
#         print self.W_hy.shape,self.h.shape , y.shape
        return y

rnn1= RNN()
rnn2 = RNN()
hidden_size = 10
input_size = 4
rnn1.setup(hidden_size,input_size)
rnn2.setup(hidden_size,input_size)

# In every step hidden states get updated

#two time steps

x = np.asarray([1,0,0,1])

y1 = rnn1.step(x)
y2 = rnn2.step(y1)
print x, y2

x = y2
y1 = rnn1.step(x)
y2 = rnn2.step(y1)
print x, y2

#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
from random import shuffle
import tensorflow as tf
tf.reset_default_graph()
pow2= 16

train_input = ['{0:016b}'.format(i) for i in range(2**pow2)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*(pow2+1))
    temp_list[count]=1
    train_output.append(temp_list)

print train_input[0]
print train_output[0]

NUM_EXAMPLES = 10000
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

tf.reset_default_graph()
print "test and training data loaded"


data = tf.placeholder(tf.float32, [None, pow2,1]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, (pow2+1)])
num_hidden = 24
num_layers=2
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

# init_op = tf.initialize_all_variables()
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

batch_size = 200
no_of_batches = int(len(train_input)) / batch_size
epoch = 200
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    if i%100 ==0:
        incorrect = sess.run(error,{data: inp, target: out})
        print "Epoch {} error: {}".format(i,incorrect*100)
incorrect = sess.run(error,{data: test_input, target: test_output})



print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

tt = np.random.randint(0,2,size=pow2)[np.newaxis,:,np.newaxis] 
p = sess.run(tf.argmax(prediction, 1),{data:tt})
print np.sum(tt),p[0]

sess.close()

# https://gist.github.com/nivwusquorum/b18ce332bde37e156034e5d3f60f8a23

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
tf.reset_default_graph()
# map_fn = tf.python.functional_ops.map_fn
map_fn = tf.map_fn

################################################################################
##                           DATASET GENERATION                               ##
##                                                                            ##
##  The problem we are trying to solve is adding two binary numbers. The      ##
##  numbers are reversed, so that the state of RNN can add the numbers        ##
##  perfectly provided it can learn to store carry in the state. Timestep t   ##
##  corresponds to bit len(number) - t.                                       ##
################################################################################

def as_bytes(num, final_size):
    res = []
    for _ in range(final_size):
        res.append(num % 2)
        num //= 2
    return res

def generate_example(num_bits):
    a = random.randint(0, 2**(num_bits - 1) - 1)
    b = random.randint(0, 2**(num_bits - 1) - 1)
    res = a + b
    return (as_bytes(a,  num_bits),
            as_bytes(b,  num_bits),
            as_bytes(res,num_bits))

def generate_batch(num_bits, batch_size):
    """Generates instance of a problem.
    Returns
    -------
    x: np.array
        two numbers to be added represented by bits.
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is one of [0,1] depending for first and
                second summand respectively
    y: np.array
        the result of the addition
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is always 0
    """
    x = np.empty((num_bits, batch_size, 2))
    y = np.empty((num_bits, batch_size, 1))

    for i in range(batch_size):
        a, b, r = generate_example(num_bits)
        x[:, i, 0] = a
        x[:, i, 1] = b
        y[:, i, 0] = r
    return x, y


################################################################################
##                           GRAPH DEFINITION                                 ##
################################################################################

INPUT_SIZE    = 2       # 2 bits per timestep
RNN_HIDDEN    = 20
OUTPUT_SIZE   = 1       # 1 bit per timestep
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01

USE_LSTM = True

inputs  = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)


if USE_LSTM:
    num_layers=2
    cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
else:
    cell = tf.nn.rnn_cell.BasicRNNCell(RNN_HIDDEN)

# Create initial state. Here it is just a constant tensor filled with zeros,
# but in principle it could be a learnable parameter. This is a bit tricky
# to do for LSTM's tuple state, but can be achieved by creating two vector
# Variables, which are then tiled along batch dimension and grouped into tuple.
batch_size    = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)

# Given inputs (time, batch, input_size) outputs a tuple
#  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
#  - states:  (time, batch, hidden_size)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)


# project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
# an extra layer here.
final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

# apply projection to every timestep.
predicted_outputs = map_fn(final_projection, rnn_outputs)

# compute elementwise cross entropy.
error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)

# optimize
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

# assuming that absolute difference between output and correct answer is 0.5
# or less we can round it to the correct output.
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))


################################################################################
##                           TRAINING LOOP                                    ##
################################################################################

NUM_BITS = 10
ITERATIONS_PER_EPOCH = 100
BATCH_SIZE = 16

valid_x, valid_y = generate_batch(num_bits=NUM_BITS, batch_size=100)

session = tf.Session()
# For some reason it is our job to do this:
session.run(tf.global_variables_initializer())

for epoch in range(200):
    epoch_error = 0
    for _ in range(ITERATIONS_PER_EPOCH):
        # here train_fn is what triggers backprop. error and accuracy on their
        # own do not trigger the backprop.
        x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
        epoch_error += session.run([error, train_fn], {
            inputs: x,
            outputs: y,
        })[0]
    epoch_error /= ITERATIONS_PER_EPOCH
    valid_accuracy = session.run(accuracy, {
        inputs:  valid_x,
        outputs: valid_y,
    })
    if epoch%10==0:
        print ("Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0))
# 

preds_valid = session.run(predicted_outputs, {
        inputs:  valid_x,
        outputs: valid_y,
    })

i = 20
print (np.around(preds_valid)[:,i,0])
print (valid_y[:,i,0])

session.close()

# From: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb

'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

tf.reset_default_graph()

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)



# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# data = tf.placeholder(tf.float32, [None, pow2,1]) #Number of examples, number of input, dimension of each input


# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}




# Prepare data shape to match `rnn` function requirements

num_layers = 2
lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden,state_is_tuple=True)
lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)
    
pred = tf.matmul(last, weights['out']) + biases['out']

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))





# Launch the graph
sess1 = tf.Session()
# For some reason it is our job to do this:
sess1.run(tf.global_variables_initializer())


# with tf.Session() as sess:
#     sess.run(init)
step = 1
    # Keep training until reach max iterations
while step * batch_size < training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # Reshape data to get 28 seq of 28 elements
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    # Run optimization op (backprop)
    sess1.run(optimizer, feed_dict={x: batch_x, y: batch_y})
    if step % display_step == 0:
        # Calculate batch accuracy
        acc = sess1.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        # Calculate batch loss
        loss = sess1.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +               "{:.6f}".format(loss) + ", Training Accuracy= " +               "{:.5f}".format(acc))
    step += 1
print("Optimization Finished!")

# Calculate accuracy for 128 mnist test images
test_len = 128
test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
test_label = mnist.test.labels[:test_len]
print("Testing Accuracy:",       sess1.run(accuracy, feed_dict={x: test_data, y: test_label}))

