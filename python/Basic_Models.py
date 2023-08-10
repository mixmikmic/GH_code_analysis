import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
MNIST = input_data.read_data_sets("../data/mnist", one_hot = True)

with tf.name_scope('input'):
    X = tf.placeholder (tf.float32, [None, 784])
    Y = tf.placeholder (tf.float32, [None, 10])

with tf.name_scope('network'):
    W1 = tf.Variable(tf.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[1, 256]))
    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.truncated_normal([256, 256], stddev = 0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[1, 256]))
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

    W_out = tf.Variable(tf.truncated_normal([256, 10], stddev = 0.1))
    b_out = tf.Variable(tf.constant(0.1, shape=[1, 10]))
    logits = tf.matmul(layer2, W_out) + b_out

with tf.name_scope('cross_entropy_loss'):
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=Y)
    loss = tf.reduce_mean(entropy)

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

Y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(Y_pred, 1)
y_cls = tf.argmax(Y, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, y_cls), tf.float32))

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
tf.summary.histogram('Weights_1', W1)
tf.summary.histogram('Bias_1', b1)
tf.summary.histogram('Weights_2', W2)
tf.summary.histogram('Bias_2', b2)
tf.summary.histogram('Weights_out', W_out)
tf.summary.histogram('Bias_out', b_out)

summary_op = tf.summary.merge_all()

batch_size = 128
epochs = 25
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('logs/train', graph=tf.get_default_graph())
n_batches = (int) (MNIST.train.num_examples/batch_size)
for i in range(epochs):
    total_loss = 0
    for batch in range(n_batches):
        X_batch, y_batch = MNIST.train.next_batch(batch_size)
        o, l, summary = sess.run([optimizer, loss, summary_op], feed_dict={X: X_batch, Y: y_batch})
        total_loss += l
        writer.add_summary(summary, i*n_batches + batch)
    print("Epoch {0}: {1}".format(i, total_loss))
    if i % 5 == 0 and i!= 0:
        X_val, y_val = MNIST.validation.next_batch(MNIST.validation.num_examples)
        val_accuracy = sess.run(accuracy, feed_dict={X: X_val, Y: y_val})
        print("\tVal Accuracy {0}".format(val_accuracy))

print("Computing accuracy ...")
X_batch, y_batch = MNIST.test.next_batch(MNIST.test.num_examples)
final_accuracy = sess.run(accuracy, feed_dict={X: X_batch, Y: y_batch})

print ("Test Accuracy {0}".format(final_accuracy))



