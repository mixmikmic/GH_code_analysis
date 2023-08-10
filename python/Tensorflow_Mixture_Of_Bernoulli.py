import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
import numpy as np

mnist = input_data.read_data_sets('MNIST_data/')
train = mnist.train.images
test  = mnist.test.images

# in place binarization without copy
train[train > 0.5] = 1. ; train[train <= 0.5] = 0.
test[test > 0.5] = 1. ; test[test <= 0.5] = 0.

# feel free to tweak the hyperparameters
lr = 1e-4                                      # learning rate for optimizer
categories = 10                                # number of categories for latent z
epochs = 10                                    # number of epochs for training
batch_size = 128                               # batch size
train_iters = train.shape[0] // batch_size     # number of iterations each epoch
log_every = 1                                  # print training status every such epochs

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
theta = tf.get_variable(name='theta', shape=[categories, 784])

# TODO: define loss function here according to equation (2) above
# Hint: you might find tf.reduce_logsumexp useful
loss = tf.reduce_sum(theta)  # this is just a stub, you should modify this

train_op = tf.train.AdamOptimizer(lr).minimize(-loss)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    
    for e in range(epochs):
        for i in range(train_iters):
            batch = train[i*batch_size:(i+1)*batch_size]
            sess.run(train_op, feed_dict={x: batch})
        
        if e % log_every == 0:
            stats = sess.run(-loss, feed_dict={x: test})
            print ('aggregate marginal log-likelihood on test set %.4f' % stats)



