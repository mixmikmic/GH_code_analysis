# load libraries
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow import nn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


# load data
# with one_hot=True, categorical labels (e.g. {1,2,3...}) coded into binary vector form (e.g.{[1,0,0],[0,1,0]...})
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# visual check for the loaded dataset
disp_target = 4
plt.imshow(mnist.train.images[disp_target].reshape((28,28)),cmap='gray_r')
plt.title(
    'Train Observation {:d} - Label : {:d}'.format(
        disp_target,
        np.argmax(mnist.train.labels[disp_target])
    )
)

# make a inference expression w/ Variables
x = tf.placeholder(tf.float32, shape=[None,28*28])
y = tf.placeholder(tf.float32, shape=[None,10])

# 3 hidden layers
h1 = tf.contrib.layers.relu(x,100)
h2 = tf.contrib.layers.relu(h1,100)
h3 = tf.contrib.layers.relu(h2,100)

# read out layer
o = tf.contrib.layers.linear(h3,10)

# EVEN MORE SIMPLER VERSION! (!CAUTION! over simplified)
# relu_lyr = tf.contrib.layers.relu # alias
# lin_lyr = tf.contrib.layers.linear # alias
# 
# o = lin_lyr(relu_lyr(relu_lyr(relu_lyr(x,100),100),100),10)
#

# set cross_entropy loss function with y (target) and o (logits)
cross_entropy = tf.reduce_mean(
    nn.softmax_cross_entropy_with_logits(
        labels=y,
        logits=o
    )
)

# then we'll get update rule (train op) here
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cross_entropy)

# binary accuracy calculation
correct_prediction = tf.equal(tf.argmax(o,1), tf.argmax(y,1))
# ...and its mean per feed
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

m = 100 # batch size
n = 1000 # number of batches to train
acc_tr = []
acc_vl = []

# Assume that you have 2GB of GPU memory and want to allocate 512MB:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

# open an session
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    
    # Initialize Variables
    sess.run(tf.global_variables_initializer())

    for j in xrange(1000):
        # here we fetch a batch from dataset
        batch = mnist.train.next_batch(m)
        # update is conducted by tf
        # feed dict is key-value dictionary which
        # each keys are placeholder and
        # each values are actual value for placeholders
        sess.run(train_step,feed_dict={x:batch[0], y:batch[1]})
        
        if j%100==0:
            acc_tr.append(accuracy.eval(feed_dict={x:batch[0], y:batch[1]}))
            acc_vl.append(accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))

            print(
                '{:d}th Iteration - Train Acc: {:.2%} - Test Acc: {:.2%}'.format(
                    j,acc_tr[-1],acc_vl[-1]
                )
            )


fig,ax = plt.subplots()
ax.plot(acc_tr, label='Training Accuracy')
ax.plot(acc_vl, label='Validation Accuracy')
ax.grid(True)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Iteration')
ax.set_title('Training Result')
ax.legend()



