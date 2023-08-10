get_ipython().magic('matplotlib inline')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/home/project/imagerecognition', validation_size = 0)

img = mnist.train.images[2]
plt.imshow(img.reshape((28, 28)), cmap = 'Greys_r')

encoding_dim = 32

image_size = mnist.train.images.shape[1]

inputs = tf.placeholder(tf.float32, (None, image_size))

w1 = tf.Variable(tf.random_normal([image_size, encoding_dim], stddev = 0.5))
b1 = tf.Variable(tf.random_normal([encoding_dim], stddev = 0.1))
encoded = tf.nn.sigmoid(tf.matmul(inputs, w1) + b1)

w2 = tf.Variable(tf.random_normal([encoding_dim, image_size], stddev = 0.5))
b2 = tf.Variable(tf.random_normal([image_size], stddev = 0.1))

logits = tf.matmul(encoded, w2) + b2
decoded = tf.nn.sigmoid(logits)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = inputs, logits = logits))
opt = tf.train.RMSPropOptimizer(0.01).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

EPOCH = 30
BATCH_SIZE = 500
for i in xrange(EPOCH):
    TOTAL_LOST = 0
    for k in xrange(mnist.train.num_examples // BATCH_SIZE):
        batch = mnist.train.next_batch(BATCH_SIZE)
        batch_cost, _ = sess.run([loss, opt], feed_dict = {inputs: batch[0]})
        TOTAL_LOST += batch_cost
    
    print 'Epoch:' + str(i + 1) + ', loss: ' + str(TOTAL_LOST / (mnist.train.num_examples // BATCH_SIZE) * 1.0)

fig, axes = plt.subplots(nrows =2, ncols =10, sharex = True, sharey = True, figsize = (20,4))
in_imgs = mnist.test.images[:10]
reconstructed = sess.run(decoded, feed_dict = {inputs: in_imgs})

for images, row in zip([in_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap = 'Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


fig.tight_layout(pad = 0.1)



