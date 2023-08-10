import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from itertools import izip

data = input_data.read_data_sets("MNIST/", one_hot=True, validation_size=0)
x_train, y_train = data.train.images, data.train.labels
x_test, y_test = data.test.images, data.test.labels

n_input = 784 # size of the input, i.e. dimension of an image 28*28 = 784
hidden_units1 = 128 # number of hidden units for the first layer
hidden_units2 = 128 # and for the second layer
n_classes = 10 # number of classes, i.e. digits from 0 to 9.

# placeholders to feed the data to the network
images = tf.placeholder(tf.float32, shape=[None, n_input])
labels = tf.placeholder(tf.int64, shape=[None, n_classes]) # e.g. digit 6 is encoder as `[0,0,0,0,0,0,1,0,0,0]`

# layer 1
w1 = tf.get_variable('weigths_1', shape=[n_input, hidden_units1],
                     initializer=tf.random_normal_initializer(stddev=1.0))
b1 = tf.get_variable('biases_1', shape=[hidden_units1])
layer1 = tf.matmul(images, w1) + b1
layer1 = tf.nn.relu(layer1) # non linearity

# layer 2
w2 = tf.get_variable('weigths_2', shape=[hidden_units1, hidden_units2],
            initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(hidden_units1)))
b2 = tf.get_variable('biases_2', shape=[hidden_units2])
layer2 = tf.matmul(layer1, w2) + b2
layer2 = tf.nn.relu(layer2) # non linearity

# output layer
w3 = tf.get_variable('weigths_3', shape=[hidden_units2, n_classes],
            initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(hidden_units2)))
b3 = tf.get_variable('biases_3', shape=[n_classes])
output = tf.matmul(layer2, w3) + b3

logits = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels)
loss = tf.reduce_mean(logits)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# calculate the accuracy
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_next_batch_iterator(batch_size):
    span = range(x_train.shape[0])
    np.random.shuffle(span)
    args = [iter(span)] * batch_size
    return izip(*args)

def get_next_batch(batch_size):
    try:
        indexes = iterator.next()
    except:
        global iterator
        iterator = get_next_batch_iterator(batch_size)
        indexes = iterator.next()
    return np.asarray([x_train[i] for i in indexes]),            np.asarray([y_train[i] for i in indexes])

steps_n = 1000
batch_size = 32
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, steps_n + 1):
        images_batch, labels_batch = get_next_batch(batch_size)
        feed_dict = {images: images_batch, labels: labels_batch}
        sess.run([optimizer], feed_dict=feed_dict)
        if i % 50 == 0:
            feed_dict = {images: x_test, labels: y_test}
            loss_, accuracy_ = sess.run([loss, accuracy], feed_dict=feed_dict)
            print('TEST: step %i, loss %.3f, accuracy %.2f' % (i, loss_, accuracy_))

