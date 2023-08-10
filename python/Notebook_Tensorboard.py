from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph) # if you prefer creating your writer using session's graph
    print(sess.run(x))
writer.close()

a = tf.constant(2, name="a")
b = tf.constant(2, name="b")
x = tf.add(a, b, name="add")
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph) # if you prefer creating your writer using session's graph
    print(sess.run(x))
writer.close()

tf.reset_default_graph()

a = tf.constant(2, name="a")
b = tf.constant(2, name="b")
x = tf.add(a, b, name="add")
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph) # if you prefer creating your writer using session's graph
    print(sess.run(x))
writer.close()

tf.reset_default_graph()

#writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.name_scope('hidden') as scope:
    a = tf.constant(5, name='alpha')
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
    b = tf.Variable(tf.zeros([1]), name='biases')

import tensorflow as tf
tf.reset_default_graph()

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)

# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("./histogram_example")

summaries = tf.summary.merge_all()

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
    k_val = step/float(N)
    summ = sess.run(summaries, feed_dict={k: k_val})
    writer.add_summary(summ, global_step=step)



