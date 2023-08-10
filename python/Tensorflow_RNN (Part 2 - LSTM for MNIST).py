import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Hyper-Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 100

n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# Define Input and Labelled Output
inputs = tf.placeholder("float", [None, n_steps, n_input])
expected_outputs = tf.placeholder("float", [None, n_classes])

# Define Variables
softmax_w = tf.Variable(tf.random_normal([n_hidden, n_classes]))
softmax_b = tf.Variable(tf.random_normal([n_classes]))

# Define Network
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

# Split raw image data into suitable sizes for sequencial input
x = tf.split(tf.reshape(tf.transpose(inputs, [1, 0, 2]), [-1, n_input]), n_steps, 0)

outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
predicted_outputs = tf.matmul(outputs[-1], softmax_w) + softmax_b

# Define Cost and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_outputs, labels=expected_outputs))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(predicted_outputs,1), tf.argmax(expected_outputs,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializer
init = tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={inputs: batch_x, expected_outputs: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={inputs: batch_x, expected_outputs: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={inputs: batch_x, expected_outputs: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +                   "{:.6f}".format(loss) + ", Training Accuracy= " +                   "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:",         sess.run(accuracy, feed_dict={inputs: test_data, expected_outputs: test_label}))

