import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data',one_hot = True)

# tf notation for permanents. None allows flexibility in that dimension
x = tf.placeholder(tf.float32,[None,784]) 

# directly go from inputs to outputs
W_linreg = tf.Variable(tf.random_normal([784, 10], stddev = 0.1)) 
b_linreg = tf.Variable(tf.random_normal([10], stddev = 0.1))

# Our prediction
y = tf.nn.softmax(tf.matmul(x,W_linreg) + b_linreg)

# correct answers
y_ = tf.placeholder(tf.float32,[None,10])

# set as negative log like
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) 
# 0.5 is the learn rate
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) 

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

## begin the training:
batch_size = 500

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={ x : batch_xs, y_: batch_ys})
    

### checking the results:
# check by comparing max values in vectors
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x : mnist.test.images, y_:mnist.test.labels}))

# tf notation for permanents. None allows flexibility in that dimension
x = tf.placeholder(tf.float32,[None,784]) 

## parameters to consider:
batch_size = 200
learn_rate = 0.5
h1_layer_size = 500

# Go from inputs to hidden layer
W_NN1 = tf.Variable(tf.random_normal([784, h1_layer_size], stddev = 0.1)) 
b_NN1 = tf.Variable(tf.random_normal([h1_layer_size], stddev = 0.1))
# Now go from hidden layer to 
W_NN2 = tf.Variable(tf.random_normal([h1_layer_size, 10], stddev = 0.1)) 
b_NN2 = tf.Variable(tf.random_normal([10], stddev = 0.1))

# Our prediction, apply some relu non-linearity to system.
h1_layer = tf.nn.relu(tf.matmul(x,W_NN1) + b_NN1) 
y = tf.nn.softmax(tf.matmul(h1_layer, W_NN2)+b_NN2)
# correct answers
y_ = tf.placeholder(tf.float32,[None,10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # set as negative log like
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # 0.5 is the learn rate

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

## begin the training:
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={ x : batch_xs, y_: batch_ys})

### checking the results:
# check by comparing max values in vectors
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x : mnist.test.images, y_:mnist.test.labels}))

# some helper function to help us set up the convolutional network
# strides : Describe how to move the convolutional window on the input
# padding : Whether to add extra zero-columns so that the window can be read to the last input column
#     'SAME' - indicates extra zero columns will be added
#     'VALID - indicates no extra columns will be added, so the number of columns is N_width - W_width 
#              in the new representation

def weight_variable(size):
    initial = tf.truncated_normal(size, stddev = 0.02)
    return tf.Variable(initial)
    
def bias_variable(size):
    initial = tf.constant(0.1, shape=size)
    return tf.Variable(initial)

def conv2d(x, W):  
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') 

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# creating the appropriate layers
# First, it's necessary to create a 4-tensor out of inputs
x = tf.placeholder(tf.float32,[None,784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # this layer will be a 28 x 28 x 32 rep.
h_pool1 = max_pool_2x2(h_conv1)     # After the pooling, we now have a 14 x 14 x 32 rep.

# creating a second layer:
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # this layer will be 28 x 28 x 64
h_pool2 = max_pool_2x2(h_conv2)      # Now it will be 7 x 7 x 64 

# Now let's create a fully connected neural network layer
W_fc1  = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# we will flatten the convolutional layers we had developed previously
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) 
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

# Last set of weights to get to the output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Output:
y_conv1 = tf.nn.softmax(tf.matmul(h_fc1, W_fc2)+b_fc2)

# begin training, this time using fancier Adam optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv1), reduction_indices=[1])) # set as negative log like
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv1,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# some training parameters:
num_iters = 2000
batch_size = 300

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(num_iters):
    batch = mnist.train.next_batch(50)
    if i%100 == 0 :
        train_accuracy = sess.run(accuracy, feed_dict={ x: batch[0], y_: batch[1]})
        print('Step %d, training accuracy %g'%(i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_:batch[1]})

num_test_set = 2000
subset_test_idx = np.random.permutation(10000)[:num_test_set]
test_subset_images = mnist.test.images[subset_test_idx]
test_subset_labels = mnist.test.labels[subset_test_idx]

#test_accuracy = sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels})
test_accuracy = sess.run(accuracy, feed_dict = {x:test_subset_images, y_:test_subset_labels})
print('Test accuracy for convnet without dropout: %g'%(test_accuracy))

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # this layer will be a 28 x 28 x 32 rep.
h_pool1 = max_pool_2x2(h_conv1)     # After the pooling, we now have a 14 x 14 x 32 rep.

# creating a second layer:
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # this layer will be 28 x 28 x 64
h_pool2 = max_pool_2x2(h_conv2)      # Now it will be 7 x 7 x 64 

# Now let's create a fully connected neural network layer
W_fc1  = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# we will flatten the convolutional layers we had developed previously
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) 
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

# Setting up parameter for dropout:
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Last set of weights to get to the output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Output:
y_conv2 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

# begin training, this time using fancier Adam optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv2), reduction_indices=[1])) # set as negative log like
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv2,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# some training parameters:
num_iters = 2000
batch_size = 300

sess.run(tf.initialize_all_variables())
for i in range(num_iters):
    batch = mnist.train.next_batch(50)
    if i%100 == 0 :
        train_accuracy = sess.run(accuracy, feed_dict={ x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('Step %d, training accuracy %g'%(i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_:batch[1], keep_prob: 0.5}) # during trainin, only keep nodes half the tiem

#test_accuracy = sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0})
test_accuracy = sess.run(accuracy, feed_dict = {x:test_subset_images, y_:test_subset_labels, keep_prob: 1.0})
print('Test accuracy for convnet without dropout: %g'%(test_accuracy))

