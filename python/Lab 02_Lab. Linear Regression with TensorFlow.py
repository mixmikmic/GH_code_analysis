import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#Our hypothesis
hypothesis = w * x_data + b

#Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#Minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#Before starting, initialize the variables.
init = tf.global_variables_initializer()

#Launch the graph
sess = tf.Session()
sess.run(init)

#Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(w), sess.run(b))

import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#Placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#Our hypothesis
hypothesis = w * X + b

#Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#Before starting, initialize the variables.
init = tf.global_variables_initializer()

#Launch the graph
sess = tf.Session()
sess.run(init)

#Fit the line
for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(w), sess.run(b))

#Learns best fit is w: [1], b: [0]
print(sess.run(hypothesis, feed_dict={X:5}))
print(sess.run(hypothesis, feed_dict={X:2.5}))



