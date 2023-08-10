from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf

# 3                                 # a rank 0 tensor; a scalar with shape []
# [1., 2., 3.]                      # a rank 1 tensor; a vector with shape [3]
# [[1., 2., 3.], [4., 5., 6.]]      # a rank 2 tensor; a matrix with shape [2, 3]
# [[[1., 2., 3.]], [[7., 8., 9.]]]  # a rank 3 tensor with shape [2, 1, 3]

a = tf.constant(3, name='const3')
b = tf.constant(4, name='const4')
x = tf.add(a, b, name='Add')
print(x)

# start a session
sess = tf.Session()

# fetch the values of Add
print(sess.run(x)) 

# close session
sess.close()

W = tf.Variable([.3], dtype=tf.float32, trainable=True, name="W") 
b = tf.Variable([-.3], dtype=tf.float32, name="b")    # trainable=True is default
b2 = tf.Variable([-.3])     # default name will be given, default dtype is tf.float32

#with tf.Session() as sess:
#    print(sess.run(W))

with tf.Session() as sess:
    
    # initialize variable
    sess.run(W.initializer)
    print(sess.run(W))
    
    # Example of assigning anew value
    sess.run(W.assign([20]))

    # You can also get a variable’s value from tf.Variable.eval()
    print(W.eval())

    # if you run initializer again, then it will overwrite the value to the initial value
    sess.run(W.initializer)
    print(W.eval())

with tf.Session() as sess:
    
    # create a initializer op
    init = tf.global_variables_initializer() 

    # run initializer op
    sess.run(init)
    
    # rest of your program
    # blah blah

a = tf.placeholder(tf.float32) 
b = tf.placeholder(tf.float32) 
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

with tf.Session() as sess:
    
    # can feed in a single value
    print(sess.run(fetches=adder_node, feed_dict={a: 3, b: 4.5}))
    
    # or a list 
    print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
    
    # or a numpy array 
    print(sess.run(adder_node, {a: np.array([1, 3]), b: np.array([2, 4])}))

# dataset: univariate features with 5 samples and a corresponding continuous label
x_data = np.array([0, 1, 2, 3, 4])
y_data = np.array([-1.1, 0.0, 2.2, 3.2, 4.8])

# plot the relationship
plt.figure()
plt.scatter(x_data, y_data, s=100)
plt.xlabel('number of beers', fontsize=20)
plt.ylabel('happiness (arb. units)', fontsize=20)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
writer.close()

tf.reset_default_graph()

# placeholder to input the data
x = tf.placeholder(tf.float32, name="features")

# create variables for the model parameters
W = tf.Variable([.3], dtype=tf.float32, name="weight") 
b = tf.Variable([-.3], dtype=tf.float32, name="bias") 

# model
linear_model = W*x + b
# linear_model = tf.add(tf.multiply(W, x), b)

# get predictions
with tf.Session() as sess:

    # initialize variables
    sess.run(tf.global_variables_initializer())
    
    # get predictions
    predictions = sess.run(linear_model, {x: x_data})
    
# plot the data and the predictions
plt.figure()
plt.scatter(x_data, y_data, s=100)
plt.xlabel('number of beers', fontsize=20)
plt.ylabel('happiness (arb. units)', fontsize=20)
plt.plot(x_data, predictions, 'r');

# create placeholder for y
y = tf.placeholder(tf.float32) 

# sum over the squared errors
loss = tf.reduce_sum(tf.square(linear_model - y) ) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) 

# create op to calculate derivatives and update parameters
train_step = optimizer.minimize(loss) 

# start session
sess = tf.Session()

# initialize all variables
sess.run(tf.global_variables_initializer()) 

# execute optimizer 
sess.run(train_step, {x: x_data, y: y_data}) 

# get predictions
predictions = sess.run(linear_model, {x: x_data})

# plot the data and the predictions
plt.figure()
plt.scatter(x_data, y_data, s=100)
plt.xlabel('number of beers', fontsize=20)
plt.ylabel('happiness (arb. units)', fontsize=20)
plt.plot(x_data, predictions, 'r');

# let's train over many epochs
for i in range(1000):   

    # execute optimizer 
    sess.run(train_step, {x: x_data, y: y_data}) 

# get predictions
predictions = sess.run(linear_model, {x: x_data})

# plot the data and the predictions
plt.figure()
plt.scatter(x_data, y_data, s=100)
plt.xlabel('number of beers', fontsize=20)
plt.ylabel('happiness (arb. units)', fontsize=20)
plt.plot(x_data, predictions, 'r');



