import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import tempfile
from Data_Processing import load_data, one_hot_encode
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos] # if x.device_type == 'GPU']

print(get_available_gpus())

parent_dir = "./UrbanSound8K/audio/"
file_title = "delta"
train_folds = np.array(range(1,9)) #first 8 folds as training set
dev_folds = np.array([9]) #9th fold as dev set
test_folds = np.array([10]) #10th fold as test set

train_pd, dev_pd, test_pd = load_data(parent_dir, file_title, train_folds, dev_folds, test_folds)

print(train_pd.shape)
print(dev_pd.shape)
print(test_pd.shape)
train_pd.head()


train_x, train_y = train_pd.iloc[:, 0:1640].values, train_pd.iloc[:, 1640].values
dev_x, dev_y = dev_pd.iloc[:, 0:1640].values, dev_pd.iloc[:, 1640].values
test_x, test_y = test_pd.iloc[:, 0:1640].values, test_pd.iloc[:, 1640].values

#use part of the training sets to save training time, use the full datasets when have enough computing power
train_x, train_y = train_x[0:12800, :], train_y[0:12800]




def weight_variable(shape, name):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)


def bias_variable(shape, name):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = name)

training_iters = 20000
lr = 2e-3
batch_size = 256
hidden_dense = 1000

def urban_cnn(x): #, is_training = True) #, sess):
    
    x_cnn = tf.reshape(x, [-1, 20, 41, 2])
    keep_prob = tf.placeholder(tf.float32)
    """
    The first convolutional ReLU layer consisted of 80 filters
    of rectangular shape (17×6 size, 1×1 stride) allowing
    for slight frequency invariance. Max-pooling was applied
    with a pool shape of 1×3 and stride of 1×3.
    """
    with tf.variable_scope("cov_1", reuse=tf.AUTO_REUSE) as scope:
  
        W_conv1 = weight_variable([17, 6, 2, 80], name = "W_Conv1")
        #scope.reuse_variables() 
        b_conv1 = bias_variable([80], name = "W_CB1")

        wc1_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        wc1_weights = tf.get_variable(
            name="W_Conv1",
            shape = [17, 6, 2, 80],
            regularizer=wc1_regularizer
        )
        #shape [-1, 20,41,80]
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_cnn, wc1_weights, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding = "SAME") 
        #endup shape [-1, 20, 14, 80]
  
        h_conv1_drop = tf.nn.dropout(h_pool1, keep_prob= keep_prob) 

    """
    A second convolutional ReLU layer consisted of 80 filters
    (1×3 size, 1×1 stride) with max-pooling (1×3 pool size,
    1×3 pool stride)
    """
    with tf.variable_scope("cov_2", reuse=tf.AUTO_REUSE) as scope_2:
        W_conv2 = weight_variable([1, 3, 80, 80], name = "W_Conv2")
        #scope_2.reuse_variables()         
        b_conv2 = bias_variable([80], name = "W_CB2")

        
        wc2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        wc2_weights = tf.get_variable(
            name="W_Conv2",
            shape = [1, 3, 80, 80],
            regularizer= wc2_regularizer
        )
        #shape [-1, 20, 14, 80]
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1_drop, wc2_weights, strides = [1, 1, 1, 1], padding="SAME") + b_conv2)
        #endup shape [-1, 20, 5, 80]
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding = "SAME") 
        h_conv2_drop = tf.nn.dropout(h_pool2, keep_prob= keep_prob) 

    """
    Further processing was applied through two fully connected hidden layers of 
    1000 ReLUs each and a softmax output layer.
    """ 
    
    with tf.variable_scope("dense_1", reuse=tf.AUTO_REUSE) as scope:
         
        W_fc1 = weight_variable([20 * 5 * 80, hidden_dense], name = "W_fc1")
        #scope.reuse_variables()
        b_fc1 = bias_variable([hidden_dense], name = "B_fc1")
        
        
        wfc1_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        wfc1_weights = tf.get_variable(
            name="W_fc1",
            shape = [20 * 5 * 80, hidden_dense], 
            regularizer=wfc1_regularizer
        )
        
        h_conv2_drop_flat = tf.reshape(h_conv2_drop, [-1, 20*5*80])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_drop_flat, wfc1_weights) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob = keep_prob)
    
    with tf.variable_scope("dense_2", reuse=tf.AUTO_REUSE) as scope:
    
        W_fc2= weight_variable([hidden_dense, hidden_dense], name = "W_fc2")
        b_fc2 = bias_variable([hidden_dense], name = "B_fc2")
        #scope.reuse_variables() 
        wfc2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        wfc2_weights = tf.get_variable(
            name="W_fc2",
            shape = [hidden_dense, hidden_dense],
            regularizer=wfc2_regularizer
        )

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, wfc2_weights) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob = keep_prob)
    
    with tf.variable_scope("dense_label", reuse=tf.AUTO_REUSE) as scope:
        W_fc3 = weight_variable([hidden_dense, 10], name = "W_fc3")
        b_fc3 = bias_variable([10], name = "B_fc3")

        y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    
    return y_conv, keep_prob


def testing(session, correct_num, test_x, test_y):
    l_range = test_x.shape[0] // batch_size if test_x.shape[0] % batch_size else test_x.shape[0] // batch_size + 1
    
    correct = 0
    
    for i in range(l_range):
        offset = (i * batch_size) 
        
        test_bx = test_x[offset: (offset + batch_size), :]
        test_by = test_y[offset: (offset + batch_size)]
        
        correct += session.run(correct_num, feed_dict = {x : test_bx, y_label : test_by, keep_prob: 1})
        
    return float(correct)/test_x.shape[0]

x = tf.placeholder(tf.float32, [None, 20 * 41 * 2], name = "input_x")

# Define loss and optimizer
y_label = tf.placeholder(tf.int32, [None], name = "input_y_label")
y_ = tf.one_hot(indices=tf.cast(y_label, tf.int32), depth=10)

# Build the graph for the deep net
y_conv, keep_prob = urban_cnn(x)

with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
    cross_entropy_sum = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                        logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy_sum)

with tf.variable_scope('adam_optimizer', reuse=tf.AUTO_REUSE):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    correct_num = tf.reduce_sum(correct_prediction)
    batch_accuracy = tf.reduce_mean(correct_prediction)

graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for itr in range(training_iters):    
        offset = (itr * batch_size) % (train_x.shape[0] - batch_size)
        batch_x = train_x[offset:(offset + batch_size), :]
        batch_y = train_y[offset:(offset + batch_size)]
        
        _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_label: batch_y, keep_prob: 0.5})
        
        if((itr+1)%300 ==0):
            train_accuracy = batch_accuracy.eval(feed_dict={
                x: batch_x, y_label: batch_y, keep_prob: 1.0})
            print("Iter " + str(itr) + ", Minibatch Loss= " +                   "{:.6f}".format(loss))
            print('step %d, training batch accuracy %g' % (itr, train_accuracy))
        if((itr+1)%1000==0):
            test_accuracy = testing(sess, correct_num, test_x, test_y)
            print("Iter " + str(itr) + ", Test Accuracy= " +                   "{:.5f}".format(test_accuracy))

sess.close()













