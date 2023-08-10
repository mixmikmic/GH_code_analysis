get_ipython().magic('matplotlib inline')

import tensorflow as tf
import numpy as np

# load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = np.reshape(mnist.train.images, [-1, 28, 28, 1])
train_labels = mnist.train.labels

test_images = np.reshape(mnist.test.images, [-1, 28, 28, 1])
test_labels = mnist.test.labels

# training input queue with large batch size
batch_size = 7500
image, label = tf.train.slice_input_producer([train_images, train_labels])
image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size)

from partialflow import GraphSectionManager

sm = GraphSectionManager()

from BasicNets import BatchnormNet

# flag for batch normalization layers
is_training = tf.placeholder(name='is_training', shape=[], dtype=tf.bool)
net = BatchnormNet(is_training, image_batch)

# first network section with initial convolution and three residual blocks
with sm.new_section() as sec0:
    with tf.variable_scope('initial_conv'):
        stream = net.add_conv(net._inputs, n_filters=16)
        stream = tf.Print(stream, [stream], 'Forward pass over section 0')
        stream = net.add_bn(stream)
        stream = tf.nn.relu(stream)
    
    with tf.variable_scope('scale0'):
        for i in range(3):
            with tf.variable_scope('block_%d' % i):
                stream = net.res_block(stream)

                
# second network section strided convolution to decrease the input resolution
with sm.new_section() as sec1:
    with tf.variable_scope('scale1'):
        stream = tf.Print(stream, [stream], 'Forward pass over section 1')
        stream = net.res_block(stream, filters_factor=2, first_stride=2)
        for i in range(2):
            with tf.variable_scope('block_%d' % i):
                stream = net.res_block(stream)

# third network section
with sm.new_section() as sec2:
    with tf.variable_scope('scale2'):
        stream = tf.Print(stream, [stream], 'Forward pass over section 2')
        stream = net.res_block(stream, filters_factor=2, first_stride=2)
        for i in range(4):
            with tf.variable_scope('block_%d' % i):
                stream = net.res_block(stream)
        
# fourth network section with final pooling and cross-entropy loss
with sm.new_section() as sec3:
    with tf.variable_scope('final_pool'):
        stream = tf.Print(stream, [stream], 'Forward pass over section 3')
        # global average pooling over image dimensions
        stream = tf.reduce_mean(stream, axis=2)
        stream = tf.reduce_mean(stream, axis=1)
        
        # final conv for classification
        stream = net.add_fc(stream, out_dims=10)
    
    with tf.variable_scope('loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits(stream, label_batch)
        loss = tf.reduce_mean(loss)

opt = tf.train.AdamOptimizer(learning_rate=0.0001)

sm.add_training_ops(opt, loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), verbose=True)

sm.prepare_training()

sec2.get_tensors_to_feed()

sec3.get_tensors_to_cache()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
_ = tf.train.start_queue_runners(sess=sess)

sess.run(loss, feed_dict={is_training: True})

sm.run_full_cycle(sess, fetches=loss, basic_feed={is_training:True})

