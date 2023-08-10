import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

get_ipython().magic('matplotlib inline')

# MNIST dataset can be automatically downloaded from within TensorFlow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)    # labels are encoded as one-hot!

img = mnist.test.images[0]
# reshape array to 28x28 matrix to be able to print it as square image
plt.imshow(img.reshape([28,28]))
plt.gray()
plt.show()
#plt.savefig("mnist7.png", dpi=600, bbox_inches='tight')

sess = tf.InteractiveSession()

# Define placeholders for input data and labels (with arbitrary batch size)
x = tf.placeholder(tf.float32, shape=[None, 784])    # input as vector of size 28*28 = 784
y_ = tf.placeholder(tf.float32, shape=[None, 10])    # labels as numbers 0 - 9

# Define variables for the graph and initialize them
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Note that we can initialize the variables right away in an interactive session!
sess.run(tf.global_variables_initializer())

# Again use the simple linear regression model (with slightly more variables)
y = tf.matmul(x, W) + b

# Use softmax loss
softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

learning_rate = 0.5
steps = 2000

optimization = tf.train.GradientDescentOptimizer(learning_rate).minimize(softmax)

# run optimization for the provided number of steps
for _ in range(steps):
    batch = mnist.train.next_batch(100)
    optimization.run(feed_dict={x: batch[0], y_: batch[1]})

# Define test data to be fed into graph
test_data = { x: mnist.test.images, y_: mnist.test.labels }

prediction = tf.argmax(y, 1)
label = tf.argmax(y_, 1)

# Calculate whether a prediction is correct or not
correct_prediction = tf.equal(prediction, label)
print correct_prediction.eval(feed_dict=test_data) # not that helpful ..

# Determine fraction of correct predictions given the true label
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print accuracy.eval(feed_dict=test_data)

import random
img = random.choice(mnist.test.images)
plt.imshow(img.reshape([28,28]))
plt.gray()
plt.show()

probs = y.eval(feed_dict={x: [img]})
print(probs)
highest_value = np.argmax(probs)

# Visualize the log-probabilities
bars = plt.bar(range(10), y.eval(feed_dict={x: [img] })[0])
bars[highest_value].set_color('r')
plt.show()

