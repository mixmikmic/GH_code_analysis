from PIL import Image
import h5py as h5
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import tensorflow.contrib.keras as kr

train_dataset = h5.File('datasets/train_catvnoncat.h5')

for i in train_dataset.keys():
    print(i)

len(train_dataset['train_set_x']), train_dataset['train_set_x'][0]

Image.fromarray(train_dataset['train_set_x'][27])

test_dataset = h5.File('datasets/test_catvnoncat.h5')
Image.fromarray(test_dataset['test_set_x'][47])

def weight_variable(shape):
    init_value = tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(init_value, name="W")

def bias_variable(shape):
    init_value = tf.constant(0.1, shape=shape)
    return tf.Variable(init_value, name="bias")

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME", name="conv2d")

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooled')

# For Training Procedure; Placeholder
x_ = tf.placeholder(tf.float32, [None, 64, 64, 3], name='x_')
y_ = tf.placeholder(tf.float32, [None], name='y_') # TODO

keeper = tf.placeholder(tf.float32)
image_res = tf.reshape(x_, [-1, 64, 64, 3])

# Convolution Layers
with tf.name_scope('conv-layer-1'):
    weight_conv1 = weight_variable([5, 5, 3, 32])
    bias_conv1 = bias_variable([32])
    result_conv1 = tf.nn.relu(conv2d(image_res, weight_conv1) + bias_conv1)
    result_pooled1 = max_pool(result_conv1)
    
with tf.name_scope('conv-layer-2'):
    weight_conv2 = weight_variable([5, 5, 32, 64])
    bias_conv2 = bias_variable([64])
    result_conv2 = tf.nn.relu(conv2d(result_pooled1, weight_conv2) + bias_conv2)
    result_pooled2 = max_pool(result_conv2)
    
with tf.name_scope('conv-layer-3'):
    weight_conv3 = weight_variable([5, 5, 64, 128])
    bias_conv3 = bias_variable([128])
    result_conv3 = tf.nn.relu(conv2d(result_pooled2, weight_conv3) + bias_conv3)
    result_pooled3 = max_pool(result_conv3)

# Feed-Forward NN Layers
with tf.name_scope('ff-layer-1'):
    weight_ff1 = weight_variable([8 * 8 * 128, 1024])
    bias_ff1 = bias_variable([1024])
    result_pool3_flat = tf.reshape(result_pooled3, [-1, 8 * 8 * 128])
    result_ff1 = tf.nn.relu(tf.matmul(result_pool3_flat, weight_ff1) + bias_ff1)
    result_ff1_dropout = tf.nn.dropout(result_ff1, keeper)
    
with tf.name_scope('ff-layer-2'):
    weight_ff2 = weight_variable([1024, 1])
    bias_ff2 = bias_variable([1])
    prediction = tf.sigmoid(tf.matmul(result_ff1_dropout, weight_ff2) + bias_ff2)
    print(prediction.shape)
    print(y_.shape)
#     print(prediction.dtype)
#     tf.Print(prediction, [prediction], 'Prediction: ')

# cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, y_))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction)))
# cross_entropy = tf.reduce_mean(tf.pow(prediction - y_, 2))
cross_entropy = tf.reduce_sum(tf.pow(prediction - y_, 2))
train_error_correction = tf.train.AdamOptimizer(1e-04).minimize(cross_entropy)

# correct_prediction = tf.equal(tf.cast(tf.argmax(prediction, 1), tf.float32), y_)
correct_prediction = tf.equal(prediction, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    start_time = time.time()
    for each_time in range(30):
        session.run(train_error_correction, feed_dict={
            x_: train_dataset['train_set_x'],
            y_: train_dataset['train_set_y'],
            keeper: 0.7
        })
        print('Time %d Accuracy %lf' % (1 + each_time, session.run(accuracy, feed_dict={
            x_: test_dataset['test_set_x'],
            y_: test_dataset['test_set_y'],
            keeper: 1.0
        })))
        
    print('')
    print('Time Cost: %d' % int(time.time() - start_time))





