# First, we include imports to make this
# notebook Python 2/3 compatible.
# You might need to pip install future
from __future__ import absolute_import, division, print_function
from builtins import range

# First, we do the basic setup.
import tensorflow as tf
tf.reset_default_graph() # Just in case we're rerunning code in the notebook

# We will be training this deep neural network on MNIST,
# so let's first load the dataset.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Now let's initialize some placeholders

# Here, x is a placeholder for our input data. Since MNIST
# uses 28x28 pixel images, we "unroll" them into a 784-pixel
# long vector. The `None` indicates that we can input an
# arbitrary amount of datapoints. Thus we are saying x is a
# matrix with 784 columns and an arbitrary (to be decided 
# when we supply the data) number of rows.
x = tf.placeholder(tf.float32, [None, 784])

# We define y to be the placeholder for our *true* y's. 
# We are giving y 10 rows because each row will be a
# one-hot vector with the correct classification of the
# image.
y = tf.placeholder(tf.float32, shape=[None, 10])

# Here we make a handy function for initializing biases. 
# Note that we are returning a "Variable" - this means
# something that is subject to change during training.
# TensorFlow is actually using gradient descent to optimize
# the value of all "Variables" in our network. 
def bias_variable(shape):
    # Here we just choose to initialize our biases to 0.
    # However, this is not an agreed-upon standard and
    # some initialize the biases to 0.01 to ensure
    # that all ReLU units fire in the beginning.
    initial = tf.constant(0.00, shape=shape)
    return tf.Variable(initial)

# Let's define the first set of weights and biases (corresponding to our first layer)
# We use He initialization for the weights as good practice for when we're training
# deeper networks. Here, get_variable is similar to when we return a Variable and assign
# it, except it also checks to see if the variable already exists.

# This is: [number of input neurons, number of neurons in the first hidden layer,
# number of neurons in the second hidden layer, number of classes]
num_neurons = [784, 1280, 768, 10]

# Just store this for convenience
he_init  = tf.contrib.layers.variance_scaling_initializer()
activ_fn = tf.nn.relu 

w1 = tf.get_variable("w1", shape=[num_neurons[0], num_neurons[1]], 
                     initializer=he_init)
b1 = bias_variable([num_neurons[1]])

# Now let's define the computation that takes this layer's input and runs it through
# the neurons. Note that we use the ReLU activation function to avoid problems
# with our gradients. This line is the equivalent of saying the output of the
# first hidden layer is max(x*w1 + b1, 0).
h1 = activ_fn(tf.matmul(x, w1) + b1)

# We also apply dropout after this layer and the next. Dropout is a form of regularization
# in neural networks where we "turn off" randomly selected neurons during training.
keep_prob = tf.placeholder(tf.float32)
h1_drop = tf.nn.dropout(h1, keep_prob)

# Define the second layer, similarly to the first.
w2 = tf.get_variable("w2", shape=[num_neurons[1], num_neurons[2]], 
                     initializer=he_init)
b2 = bias_variable([num_neurons[2]])
h2 = activ_fn(tf.matmul(h1_drop, w2) + b2)
h2_drop = tf.nn.dropout(h2, keep_prob)

# And define the third layer to output the log probabilities.
# Note that this wouldn't really be considered a "deep" network
# since there's only two hidden layers, but it should be clear to
# see how hidden layers can easily be added at this point to make
# it "deep".
w3 = tf.get_variable("w3", shape=[num_neurons[2], num_neurons[3]], 
                     initializer=he_init)
b3 = bias_variable([num_neurons[3]])
logits = tf.matmul(h2_drop, w3) + b3

# We define our loss function to be cross entropy over softmax probabilities.
# Here our true labels are defined by y, and our log probabilities
# (TensorFlow calls them `logits`) are defined by logits.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
# If we wanted, we could also add L2 weight regularization by adding
# the following lines to the loss function
#     0.0001*tf.nn.l2_loss(w1) +\
#     0.0001*tf.nn.l2_loss(w2) +\
#     0.0001*tf.nn.l2_loss(w3)

# We will use the `Adam` optimizer. Adam is an fancier variant of
# standard gradient descent.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Here we build a binary vector corresponding to where our predicted 
# classes matched the actual classes.
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    num_train = mnist.train.num_examples
    num_test  = mnist.test.num_examples

    num_epochs = 20
    batch_size = 50
    
    # Train
    for i in range(num_epochs):
        for _ in range(num_train / batch_size):
            batch = mnist.train.next_batch(batch_size)
            train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
        # Get an estimate of our current progress using the last batch
        train_accuracy, loss = sess.run([accuracy, cross_entropy], 
                                    feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
        print("epoch %d completed: training accuracy %g, loss %g"%(i, train_accuracy, loss))

    # Test
    test_accuracy = 0
    for _ in range(num_test / batch_size):
        batch = mnist.test.next_batch(batch_size)
        test_accuracy += batch_size * accuracy.eval(feed_dict={
            x:batch[0], y: batch[1], keep_prob: 1.0})

    print("test accuracy %g"%(test_accuracy / num_test))

from tensorflow.contrib.layers import fully_connected, dropout, batch_norm
tf.reset_default_graph()

x  = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
num_neurons = [784, 1280, 768, 10]
he_init  = tf.contrib.layers.variance_scaling_initializer()
activ_fn = tf.nn.relu 

# Instead of making keep_prob a placeholder (like we did for dropout
# above), we can make a boolean `is_training` placeholder that dropout
# and batch normalization can check to determine what parameter
# values to use (i.e. if is_training = True, then dropout will use
# a keep_prob of 0.5. Otherwise, it uses a keep_prob of 1.0).
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

# We can even easily add Batch Normalization, which can also be quite
# useful when training deep neural networks (although it won't do much
# here).
bn_params = {
    'is_training': is_training,
    'decay': 0.99,
    'updates_collections': None
}

# Define the first hidden layer using `fully_connected`
# There are similar functions (e.g. conv2d) for other
# types of layers.
keep_prob = 0.5
hidden1 = fully_connected(x, num_neurons[1], 
                          weights_initializer=he_init,
                          activation_fn=activ_fn,
                          normalizer_fn=batch_norm, 
                          normalizer_params=bn_params)
hidden1_drop = dropout(hidden1, keep_prob, is_training=is_training)

hidden2 = fully_connected(hidden1_drop, num_neurons[2], 
                          weights_initializer=he_init,
                          activation_fn=activ_fn,
                          normalizer_fn=batch_norm, 
                          normalizer_params=bn_params)
hidden2_drop = dropout(hidden2, keep_prob, is_training=is_training)

logits = fully_connected(hidden2_drop, num_neurons[3], activation_fn=None)

# Let's train it and see how it does! It should be pretty similar
# to our previous results.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    num_train = mnist.train.num_examples
    num_test  = mnist.test.num_examples

    num_epochs = 20
    batch_size = 50
    
    # Train
    for i in range(num_epochs):
        for _ in range(num_train / batch_size):
            batch = mnist.train.next_batch(batch_size)
            train_step.run(feed_dict={x: batch[0], y: batch[1], is_training: True})
        # Get an estimate of our current progress using the last batch
        train_accuracy, loss = sess.run([accuracy, cross_entropy], 
                                    feed_dict={x:batch[0], y: batch[1], is_training: False})
        print("epoch %d completed: training accuracy %g, loss %g"%(i, train_accuracy, loss))

    # Test
    test_accuracy = 0
    for _ in range(num_test / batch_size):
        batch = mnist.test.next_batch(batch_size)
        test_accuracy += batch_size * accuracy.eval(feed_dict={
            x:batch[0], y: batch[1], is_training: False})

    print("test accuracy %g"%(test_accuracy / num_test))

# See DeepLearningTensorFlowRecitation.py

# Now, we will train another classifier for the MNIST dataset
# except this time, we will use an RNN. While this may not be
# an especially intuitive application, in my opinion, it is an
# interesting (although also not very practical) application of
# RNNs for that reason. 

# So how do we do this? Since the images are 28 x 28 pixels, we
# will model them as a sequence of 28 pixel vectors across 28 
# timesteps. We will feed each of these pixel vectors into a
# GRU cell with 150 neurons. At the end of the 28 timesteps, 
# we will take the state of the RNN and feed it into a fully
# connected layer with 10 outputs, allowing us to generate
# log probabilities for each of the classes. Then, the rest
# proceeds as above where we can do softmax and cross-entropy
# on the log probabilities to determine the loss, and use that
# loss for backpropagation through the network.

tf.reset_default_graph()

num_timesteps = 28
# [num inputs per timestep, num neurons in RNN Cell, num outputs for fully connected layer]
num_neurons = [28, 150, 10] 

# Since the input data initially comes as a 784-dimension vector,
# we need to reshape it back into a 28x28 image. Now x is a tensor
# where the first dimension indexes each image.
x = tf.placeholder(tf.float32, [None, num_timesteps, num_neurons[0]])
y = tf.placeholder(tf.float32, shape=[None, 10])

# Here is where we define the core of the network. Right now, 
# we are using a GRU cell with 150 neurons. While a basic RNN cell
# also works, using the GRU allows us to track long-term
# dependencies, which improves our accuracy here. We will then feed
# it into the dynamic_rnn function, which will run all the 
# timesteps for the RNN. Note that since we know the number of 
# timesteps for every input, we could use the static_rnn function.
# However, the dynamic_rnn function seems to be strictly better as
# it has an easier API (don't need to stack and unstack the data)
# and can even support offloading GPU memory to CPU memory to avoid
# OutOfMemory errors.
basic_cell = tf.contrib.rnn.GRUCell(num_units=num_neurons[1])
outputs, final_state = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32)

# Now we take the final state of the RNN and feed it into a
# fully connected layer to obtain our log probabilities.
logits = fully_connected(final_state, num_neurons[2], activation_fn=None)

# From here on, this code should seem familiar as it is essentially
# the same code as above.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    num_train = mnist.train.num_examples

    num_epochs = 100
    batch_size = 150

    sess.run(tf.global_variables_initializer())
    # Train
    for i in range(num_epochs):
        for _ in range(num_train / batch_size):
            batch = mnist.train.next_batch(batch_size)
            x_batch = batch[0].reshape((-1, num_timesteps, num_neurons[0]))
            sess.run(train_step, feed_dict={x: x_batch, y: batch[1]})
        train_accuracy, loss = sess.run([accuracy, cross_entropy], 
                                        feed_dict={x: x_batch, y: batch[1]})
        print("epoch %d completed: training accuracy %g, loss %g"%(i, train_accuracy, loss))
        
    # Test
    x_test = mnist.test.images.reshape((-1, num_timesteps, num_neurons[0]))
    y_test = mnist.test.labels
    test_accuracy = accuracy.eval(feed_dict={x: x_test, y: y_test})
    print("test accuracy %g"%(test_accuracy))

# Let us first download the dataset we will be using,
# the works of Shakespeare. Dataset from Andrej Karpathy.
import urllib2
print ('Downloading Shakespeare data')
source = urllib2.urlopen("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
shakespeare = source.read()
print ('Download complete')

len(shakespeare)

# First we need to generate a mapping between unique
# characters 
num_chars = len(set(shakespeare))
i2c_map = {i: c for i, c in enumerate(set(shakespeare))}
c2i_map = {c: i for i, c in i2c_map.iteritems()}

tf.reset_default_graph()

num_timesteps = 30

# [num inputs per timestep, num neurons in RNN Cell, num outputs for fully connected layer]
num_neurons = 150 # [num_chars, 150, num_chars] 
batch_size  = 1

x = tf.placeholder(tf.float32, [batch_size, None, num_chars])
y = tf.placeholder(tf.float32, shape=[None, num_chars])

state = tf.zeros((batch_size, num_neurons))
basic_cell = tf.contrib.rnn.GRUCell(num_units=num_neurons)
outputs, final_state = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32, initial_state=state)

# outputs :: [batch_size, timesteps, 150]
# logits  :: [batch_size, timesteps, num_chars]

w = tf.get_variable("w", shape=[num_neurons, num_chars])
b = bias_variable([num_chars])
logits = tf.tensordot(outputs, w, [[2], [0]]) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits,2), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training Step
import numpy as np

shakespeare_trim = shakespeare[5000:200000]
with tf.Session() as sess:
    num_train = len(shakespeare_trim)
    
    num_epochs  = 1
    current_idx = 0
    
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    rnn_state = np.load('rnn_state.npy') # tf.zeros((batch_size, num_neurons)).eval()

    # Train
    for i in range(num_epochs):
        chars_per_iter = batch_size * num_timesteps
        num_iterations = num_train / chars_per_iter
        for j in range(num_iterations):
            x_batch = shakespeare_trim[current_idx:(current_idx + chars_per_iter)]
            y_batch = shakespeare_trim[(current_idx + 1):(current_idx + chars_per_iter + 1)]
            current_idx += chars_per_iter
            x_batch = [c2i_map[c] for c in x_batch]
            x_batch = tf.reshape(tf.one_hot(x_batch, num_chars), (batch_size, num_timesteps, num_chars)).eval()
            y_batch = [c2i_map[c] for c in y_batch]
            y_batch = tf.one_hot(y_batch, num_chars).eval()
            _, rnn_state = sess.run([train_step, final_state], 
                                    feed_dict={x: x_batch, y: y_batch, state: rnn_state})
            if j % 50 == 0:
                train_accuracy, loss = sess.run([accuracy, cross_entropy], 
                                                feed_dict={x: x_batch, y: y_batch, state: rnn_state})
                print("iter %d / %d completed: training accuracy %g, loss %g"%(j, num_iterations, train_accuracy, loss))
                
    # Save the model
    save_path = saver.save(sess, "./ShakespeareRNN.ckpt")
    print("Model saved in file: %s" % save_path)
    np.save('rnn_state', rnn_state)

# Generation Step
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "./ShakespeareRNN.ckpt")
    print("Model restored.")
    rnn_state = np.load('rnn_state.npy')

    num_chars_to_generate = 50
    
    seed = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\n"
    
#     if len(seed) > 0:
    x_in = np.zeros( (1, len(seed), num_chars) )
    for i,c in enumerate(seed):
        x_in[0,i,:] = tf.one_hot(c2i_map[c], num_chars).eval().reshape(1,1,num_chars)
    output = ""
#     else:
#         x_in = rnn_state.reshape( (1, 1, -1) )
#         output = ""
    
    for _ in range(num_chars_to_generate):
        rnn_output, rnn_state = sess.run([logits, final_state], feed_dict={x: x_in, state: rnn_state})
        rnn_output = rnn_output[0][0]
        next_char_idx = tf.argmax(rnn_output, axis=0).eval()
        next_char = i2c_map[next_char_idx]
        output += next_char
        x_in = tf.one_hot(next_char_idx, num_chars).eval().reshape(1,1,num_chars)
    print(output)

shakespeare[:1]

import time
start_time = time.time()
for _ in range(10):
    time.sleep(1)
    print("Elapsed time: %d sec" % (time.time() - start_time))



