get_ipython().system(' pip show tensorflow')

#! pip install --upgrade tensorflow

import tensorflow as tf

x=tf.constant(1.0)
W=tf.constant(6.0)
b=tf.constant(1.5)

y=x*W+b

print(y)

with tf.Session() as sess:
    print(sess.run(y))

x=tf.constant(1.0)
W=tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b=tf.Variable(tf.zeros([1]))

# Before starting, initialize the variables
# new versions of tf use: 
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()

y=x*W+b

with tf.Session() as sess:
    sess.run(init)
    for step in range(4):
        print(sess.run(y))


x=tf.placeholder(tf.float32)
W=tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b=tf.Variable(tf.zeros([1]))

# Before starting, initialize the variables
# new versions of tf use: 
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()

y=x*W+b

with tf.Session() as sess:
    sess.run(init)
    for step in range(4):
        print(sess.run(y,feed_dict={x:7.0}))



