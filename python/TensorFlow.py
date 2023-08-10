import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

data_path = "./../data/MNIST/"
if not os.path.exists(data_path):
    os.makedirs(data_path)

mnist = input_data.read_data_sets(data_path, one_hot=True)

# Parameters
learning_rate = 0.001
num_iters = 1000
batch_size = 128

geometry = [28, 28]
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_classes = len(classes)
dropout_prob = 0.5

# Tensor Flow Graph Input
X = tf.placeholder(tf.float32, [None, geometry[0]*geometry[1]])
y = tf.placeholder(tf.float32, [None, num_classes])
dropout = tf.placeholder(tf.float32)

# Weight & bias
# 5x5 conv, 1 input, 32 outputs
Wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
bc1 = tf.Variable(tf.random_normal([32]))

# 5x5 conv, 32 input, 64 outputs
Wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
bc2 = tf.Variable(tf.random_normal([64]))

# Fully connected (Standard 2-layer MLP), 7*7*64 input, 1024 
Wf1 = tf.Variable(tf.random_normal([7*7*64, 1024]))
bf1 = tf.Variable(tf.random_normal([1024]))

Wf2 = tf.Variable(tf.random_normal([1024, num_classes]))
bf2 = tf.Variable(tf.random_normal([num_classes]))

# Convolution Network

# Reshape input picture
input_X = tf.reshape(X, shape=[-1, 28, 28, 1])

# Stage 1 : Convolution -> ReLU -> Max Pooling -> Dropout
conv1 = tf.nn.conv2d(input_X, Wc1, strides = [1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.relu(tf.nn.bias_add(conv1, bc1))
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
conv1 = tf.nn.dropout(conv1, dropout)

# Stage 2 : Convolution -> ReLU -> Max Pooling -> Dropout
conv2 = tf.nn.conv2d(conv1, Wc2, strides = [1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.relu(tf.nn.bias_add(conv2, bc2))
conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
conv2 = tf.nn.dropout(conv2, dropout)

# Stage 3 : Fully connected : Linear -> ReLU -> Dropout
fc1 = tf.reshape(conv2, [-1, Wf1.get_shape().as_list()[0]])
fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, Wf1), bf1))
fc1 = tf.nn.dropout(fc1, dropout)

fc2 = tf.add(tf.matmul(fc1, Wf2), bf2)

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# Launch the Graph
with tf.Session() as sess:
    sess.run(init)
    
    # Train
    for epoch in range(1, num_iters+1):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training data
        sess.run(optimizer, feed_dict={X: batch_xs, y: batch_ys, dropout: dropout_prob})
        
        if epoch % 100 == 0:
            loss = sess.run(cost, feed_dict={X: batch_xs, y: batch_ys, dropout: 1.})
            print("Epoch : ", epoch, " loss=" , loss)
    
    print("Optimization Finishied")
    
    # Test
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, 
                                                             y: mnist.test.labels, 
                                                             dropout: 1.}) )

