get_ipython().magic('matplotlib inline')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Global parameters
eta = 0.01 # learning rate
n_epochs = 4
n_input = 28
n_classes = 10
batch_size = 100
n_batches = mnist.train.images.shape[0]//batch_size

# Network parameters
n_hidden = 20 # number of hidden units per layer
n_layers = 3 # number of layers 
n_steps = 28 # number of truncated backprop steps

# Create placeholder variables for the input and targets
X_placeholder = tf.placeholder(tf.float32, shape=[batch_size, n_steps, n_input])
y_placeholder = tf.placeholder(tf.int32, shape=[batch_size, n_classes])

# Create placeholder variables for final weight and bias matrix 
V = tf.Variable(tf.random_normal(shape=[n_hidden, n_classes]))
c = tf.Variable(tf.random_normal(shape=[n_classes]))

# For each initialized LSTM cell we need to specify how many hidden
# units the cell should have.
cell = tf.contrib.rnn.LSTMCell(num_units=n_hidden)

# To create multiple layers we call the MultiRNNCell function that takes 
# a list of RNN cells as an input and wraps them into a single cell
cell = tf.contrib.rnn.MultiRNNCell([cell]*n_layers)

# Create a zero-filled state tensor as an initial state
init_state = cell.zero_state(batch_size, tf.float32)

# Create a recurrent neural network specified by "cell", i.e. unroll the
# network.
# Returns a list of all previous RNN hidden states and the final states.
# final_state contains n_layer LSTMStateTuple that contain both the 
# final hidden and the cell state of the respective layer.
outputs, final_state = tf.nn.dynamic_rnn(cell, 
                                         X_placeholder, 
                                         initial_state=init_state)

temp = tf.transpose(outputs, [1,0,2])
last_output = tf.gather(temp, int(temp.get_shape()[0]-1))

# After gathering the final activations we can easily compute the logits
# using a single matrix multiplication
logits = tf.matmul(last_output, V)+c

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder,
                                                           logits=logits))

train_step = tf.train.AdamOptimizer(eta).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_placeholder,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    
    # We first have to initialize all variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Train for the specified number of epochs
    for epoch in range(n_epochs):
        
        print()
        print("Epoch: ", epoch)
        
        for batch in range(n_batches):
            
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            x_batch = x_batch.reshape((batch_size, n_steps, n_input))
            
            _train_step = sess.run(train_step, 
                                        feed_dict=
                                        {X_placeholder:x_batch,
                                         y_placeholder:y_batch
                                        })
            
            
            if batch%100 == 0:
                _loss, _accuracy = sess.run([loss, accuracy],
                                 feed_dict={
                                     X_placeholder:x_batch,
                                     y_placeholder:y_batch
                                 })
                print("Minibatch loss: %s  Accuracy: %s" % (_loss, _accuracy))
          
    print()
    print("Optimization done! Let's calculate the test error")
    
    # Evaluate the model on the first "batch_size" test examples
    x_test_batch, y_test_batch = mnist.test.next_batch(batch_size)
    x_test_batch = x_test_batch.reshape((batch_size, n_steps, n_input))
    
    test_loss, test_accuracy, _train_step = sess.run([loss, accuracy, train_step],
                                                    feed_dict={
                                                        X_placeholder:x_test_batch,
                                                        y_placeholder:y_test_batch
                                                    })
    print()
    print("Loss on test set: ", test_loss)
    print("Accuracy on test set: ", test_accuracy)







