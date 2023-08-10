# for compatibility 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import datasets
dataset = datasets.load_iris()

# split into input data (x) and GT labels (y)
data_x = dataset.data
data_y = dataset.target
print(data_x.shape)
print(data_y.shape)

new_indices = np.random.permutation(data_y.shape[0])

train_x = data_x[new_indices[:125],:]
train_y = data_y[new_indices[:125]]

test_x = data_x[new_indices[125:],:]
test_y = data_y[new_indices[125:]]

# placeholder for data x (4 attributes), one prediction label
x    = tf.placeholder("float", shape=[None, 4])
y_GT = tf.placeholder("int64", shape=[None, ])

# model parameters
n_hidden = 100
W_h = tf.Variable(0.1*tf.random_normal([4, n_hidden]), name="W_h")
b_h = tf.Variable(tf.random_normal([n_hidden]), name="b_h")
hidden_layer = tf.matmul(x, W_h) + b_h

# model parameters
W = tf.Variable(0.1*tf.random_normal([n_hidden, 3]), name="W")
b = tf.Variable(tf.zeros([3]), name="b")

# putting the model together
z = tf.matmul(hidden_layer, W) + b
y = tf.nn.softmax(z)

print(y)

# one-hot encoding of the class labels
y_GT_one_hot  = tf.one_hot(y_GT, depth=3)

# cross-entropy loss
cross_entropy = -tf.reduce_sum(y_GT_one_hot * tf.log(y+1e-10))

# alternative implementation of cross-entropy in tf. (a bit more stable numerical)
#cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y_GT_one_hot, logits=z)

# define optimizer
opt = tf.train.GradientDescentOptimizer(0.001)
train_op = opt.minimize(cross_entropy)

cross_entropy

# create operation to inialize all variables
init_op = tf.initialize_all_variables()

# launch session
with tf.Session() as sess:
    sess.run(init_op)
    
    print(W)
    #print sess.run(W) #print W.eval()

# one step of training
with tf.Session() as sess:
    
    sess.run(init_op)
    
    # one step training -- repeat this to fully optimize
    sess.run(train_op, feed_dict={x: data_x, y_GT: data_y})
    
    xen = sess.run(cross_entropy, feed_dict={x: data_x, y_GT: data_y})
    print(xen)
    

# define the accuracy 
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_GT, tf.argmax(y, 1)), tf.float32))

# summary 
tf.scalar_summary("cross_entropy", cross_entropy)
tf.scalar_summary("accuracy", accuracy)
summary_op = tf.merge_all_summaries()

# multiple steps of training

with tf.Session() as sess:
    sess.run(init_op)
    
    summary_writer  = tf.train.SummaryWriter('tf_logs', sess.graph)
    for iterIndex in range(200):
        sess.run(train_op, feed_dict={x: train_x, y_GT: train_y})
        
        summary = sess.run(summary_op, feed_dict={x: train_x, y_GT: train_y})
        summary_writer.add_summary(summary, iterIndex)
        
        if iterIndex%10==0:
            
            xen = sess.run(cross_entropy, feed_dict={x: test_x, y_GT: test_y})
            print("Iter %3d -- Cross-entropy: %f"%(iterIndex,xen))
            
            acc = sess.run(accuracy, feed_dict={x: test_x, y_GT: test_y})
            print("Iter %3d -- Accuracy: %f"%(iterIndex,acc))

