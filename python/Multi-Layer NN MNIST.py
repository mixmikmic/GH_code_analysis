# Classic imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Placeholders
x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

# Hyperparameters
learningRate = 0.1
numHiddenUnits = 50
numIterations = 5000

# TODO: Modify this block of code (and the following block) so that we create a multiple hidden layer NN 
# instead of an NN with just a single layer

# W1 and Bias Variables
W1 = tf.Variable(tf.truncated_normal([784, numHiddenUnits], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1), [numHiddenUnits])

# W2 and Bias Variables
W2 = tf.Variable(tf.truncated_normal([numHiddenUnits, 10], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1), [10])

# Intermediate Operations
H1 = tf.nn.relu(tf.matmul(x, W1) + B1)
yPred = tf.matmul(H1, W2) + B2

# Loss Function and Optimizer
crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = yPred))
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(crossEntropyLoss)

# Help us with visualizing the accuracy as the network trains
correctPredictions = tf.equal(tf.argmax(yPred, 1), tf.argmax(y_, 1)) 
accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))

# Global Variable Init
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init) 

for i in range(numIterations):
    batch = mnist.train.next_batch(100)
    optimizer.run(feed_dict = {x: batch[0], y_: batch[1]})
    # every 100 iterations, print out the accuracy
    if i % 100 == 0:
        # accuracy and loss are both functions that take (x, y) pairs as input, and run a forward pass through the network to obtain a prediction, and then compares the prediction with the actual y.
        acc = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1]})
        loss = crossEntropyLoss.eval(feed_dict = {x: batch[0], y_: batch[1]})
        print("Epoch: {}, accuracy: {}, loss: {}".format(i, acc, loss))

# evaluate our testing accuracy       
acc = accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
print("testing accuracy: {}".format(acc))



