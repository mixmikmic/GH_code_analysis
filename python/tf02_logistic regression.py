import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30

# Step 1: Read in data
#using TF Learn's built in function to load MNIST data to the folder data/mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA', one_hot = True)

# Step 2: Create placeholders for features and labels
#each image in the MNIST data is of shape 28*28 = 784
#therefore, each image is represented with a 1*784 tensor
#there are 10 classes for each image, corresponding to digits 0 - 9
# each lable is one hot vector

X = tf.placeholder(tf.float32, [batch_size, 784], name='X_placeholder')
Y= tf.placeholder(tf.float32, [batch_size, 10], name='Y_placeholder')

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y

w = tf.Variable(tf.random_normal(shape=[784, 10], stddev= 0.01), name='weights')
tf.summary.histogram("weights",w) #可视化观看变量

b = tf.Variable(tf.zeros([1, 10]), name="bias")
tf.summary.histogram("biases",b) #可视化观看变量

# Step 4: build model
# the model that returns the logits
# this logits will be later passed through softmax layer

y_predicted = tf.matmul(X, w) + b

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function

entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits = y_predicted, name='loss')
# computes the mean over all the examples in the batch
loss = tf.reduce_mean(entropy)
tf.summary.histogram("loss",loss) #可视化观看变量

# Step 6: define training op
#using gradient descent with learning rate of 0.01 to minimize loss

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    # to visualize using TensorBoard
    
    merged = tf.summary.merge_all()  #合并到Summary中    
    writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)
    
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples/batch_size)
    # train the model n_epochs times
    for i in range(n_epochs):
        total_loss = 0
        
        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})
            total_loss += loss_batch
        print ('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
        
    print ('Total time: {0} seconds'.format(time.time() - start_time))
    
    #should be around 0.35 after 25 epochs
    print('Optimization Finished!')
    
    #test the model
    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, y_predicted], feed_dict={X: X_batch, Y: Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)
        
    print ('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))

    writer.close()



