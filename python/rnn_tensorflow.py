import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Input/Output data
data_dir = '/Volumes/Transcend/[DataArchive]/WisconsinArchives/extractTrainingData/training_data_art_mfcc_vowel_noDeriv/female'
with open(data_dir+'/acoustics.pckl', 'rb') as f:
    acoustics = pickle.load(f)
with open(data_dir+'/articulation.pckl', 'rb') as f:
    articulation = pickle.load(f)

n_examples = len(acoustics)
print(n_examples)

# Learning parameters
learning_rate = 0.001
max_iter = 1000

# Network Parameters
n_input_dim = acoustics[0].shape[0]
# n_input_len = char_input.shape[0]
n_output_dim = articulation[0].shape[0]
# n_output_len = char_output.shape[0]
n_hidden = 200
n_examples = len(acoustics)

# TensorFlow graph
# (batch_size) x (time_step) x (input_dimension)
x_data = tf.placeholder(tf.float32, [1, None, n_input_dim])
# (batch_size) x (time_step) x (output_dimension)
y_data = tf.placeholder(tf.float32, [1, None, n_output_dim])

# Parameters
weights = {
    'out': tf.Variable(tf.truncated_normal([n_hidden, n_output_dim]))
}
biases = {
    'out': tf.Variable(tf.truncated_normal([n_output_dim]))
}

def RNN(x, weights, biases):
    cell = tf.contrib.rnn.BasicRNNCell(n_hidden) # Make RNNCell
    outputs, states = tf.nn.dynamic_rnn(cell, x, time_major=False, dtype=tf.float32)
    '''
    **Notes on tf.nn.dynamic_rnn**

    - 'x' can have shape (batch)x(time)x(input_dim), if time_major=False or 
                         (time)x(batch)x(input_dim), if time_major=True
    - 'outputs' can have the same shape as 'x'
                         (batch)x(time)x(input_dim), if time_major=False or 
                         (time)x(batch)x(input_dim), if time_major=True
    - 'states' is the final state, determined by batch and hidden_dim
    '''
    
    # outputs[-1] is outputs for the last example in the mini-batch
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x_data, weights, biases)
cost = tf.reduce_mean(tf.squared_difference(pred, y_data))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Learning
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_iter):
        for n in range(n_examples):
            x_train = acoustics[n].reshape((1, acoustics[n].shape[1], n_input_dim))
            y_train = articulation[n].reshape((1, articulation[n].shape[1], n_output_dim))
            _, loss, p = sess.run([optimizer, cost, pred],
                                  feed_dict={x_data: x_train, y_data: y_train})
        if (i+1) % 1 == 0:
            print('Epoch:{:>4}/{},'.format(i+1,max_iter),
                  'Cost:{:.4f},'.format(loss))

