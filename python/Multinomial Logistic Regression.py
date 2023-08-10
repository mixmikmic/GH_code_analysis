get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import numpy as np

# To be compatible with python3
try:
    import cPickle as pickle
except ImportError:
    import pickle

import gzip
import time
import tensorflow as tf
import sys

print(sys.version_info)
print(tf.__version__)

with gzip.open('../../lasagne/mnist_4000.pkl.gz', 'rb') as f:
    (X,y) = pickle.load(f, encoding='latin1')
PIXELS = len(X[0,0,0,:])
print(X.shape, y.shape, PIXELS) #As read
fig = plt.figure(figsize=(10,30))
for i in range(3):
    a=fig.add_subplot(1,3,(i+1))
    plt.imshow(-X[i,0,:,:], interpolation='none',cmap=plt.get_cmap('gray'))

# We need to reshape for the logistic regression
X = X.reshape([4000, 784])
np.shape(X)

# Taken from http://stackoverflow.com/questions/29831489/numpy-1-hot-array
def convertToOneHot(vector, num_classes=None):
    result = np.zeros((len(vector), num_classes), dtype='float32')
    result[np.arange(len(vector)), vector] = 1
    return result

convertToOneHot(y[0:3], 10)

tf.reset_default_graph()
tf.set_random_seed(1)
#Note that we usually do not like to specify the batchsize. Choosing it `None` will leave it open
x = tf.placeholder(tf.float32, shape=[None, 784], name='x_data')
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_data')

# We start with random weights a
w = tf.Variable(tf.random_normal([784, 10], stddev=0.01))
b = tf.Variable(tf.zeros([10]))

#<-------------------------- Your code here ---------------
# Your code here, do a matrix multiplication between x,w and an addtion of b
z = tf.matmul(x,w) + b
# End of your code

out = tf.nn.softmax(z)
init_op = tf.global_variables_initializer() 

tf.summary.FileWriter("/tmp/dumm/mlp_tensorflow_solution/", tf.get_default_graph()).close() #<--- Where to store

with tf.Session() as sess:
    sess.run(init_op)
    res_val = sess.run(out, feed_dict={x:X[0:2]})
res_val

loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(out), reduction_indices=[1]))

#train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train_op = tf.train.AdagradOptimizer(0.1).minimize(loss)
init_op = tf.global_variables_initializer() 
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(1000):
        idx = np.random.permutation(2400)[0:128] #Easy minibatch of size 64
        #res, out_val, _ = sess.run((loss, out, train_op),feed_dict={x:X[idx], y_true:convertToOneHot(y[idx], 10)})
        loss_, out_val, _ = sess.run((loss, out, train_op),feed_dict={x:X[idx], y_true:convertToOneHot(y[idx], 10)})
        if (i % 100 == 0):
            print(loss_)
    
    # Get the loss for the validation results (from 2400:3000)
    print('Loss for the validation set')
    #<-------------------------- Your code here ---------------
    loss_val = sess.run((loss), feed_dict={x:X[2400:3000], y_true:convertToOneHot(y[2400:3000], 10)})
    print(loss_val)
    # Get the results for the validation set
    res_val = sess.run((out), feed_dict={x:X[2400:3000]})
    #<-------------------------  End of your code here --------

# and estimate the preformance on the validation set
# Your code here
np.mean(np.argmax(res_val, axis = 1) == y[2400:3000])

