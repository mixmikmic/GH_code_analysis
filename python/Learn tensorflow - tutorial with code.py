# Import the 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None,10])

# What is this x? Some sort of placeholder... 
# Not an actual value.
x

x._shape

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# This gives us an error, 
# since the weights are not yet initialized
W.eval()

# Initialize the variables. Defaults to 0's.

# Does not work on older TensorFlow versions.
#sess.run(tf.global_variables_initializer())

# Use this line for older tensorflow versions.
sess.run(tf.initialize_all_variables())

# Yay! We can now see the (all zero) weights.
W.eval()

# This is a numpy array now.
tmp = W.eval()
print(type(tmp))
print(W.eval().shape)

# Check out the b values.
b.eval()

# This does not work... as might be expected, 
# we cannot assign this way.
b.eval()[0] = 1
b.eval()

# 
y = tf.matmul(x,W) + b

# This does not work, since nothing has been evaluated yet.
# y is just a placeholder for now.
y.eval()

errors = tf.nn.softmax_cross_entropy_with_logits(y,y_)

cross_entropy = tf.reduce_mean(errors)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

batch = mnist.train.next_batch(100)

print(type(batch))
# Tuples are like lists, but tupples cannot be changed.

# Each item in the tupple is a numpy array.
print(type(batch[0]))

# This must be the data... 
# i.e., 100 images by (28*28)=784
batch[0].shape

# This must be class labels. 100 images each with 10
# possible classes.
batch[1].shape

# Nothing here.
batch[2].shape

import numpy as np

np.sqrt(784)

idx = 0
img = np.reshape(batch[0][idx], [28,28])

img.shape

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline # So the figures are shown inline.')

# What is this..? Looks like a messed up number.
plt.imshow(img, interpolation="none")

# Check the class label.
# Okay... this should be a 7 I guess? 
# Counting starts at 0...7
batch[1][idx]

idx = 2
img = np.reshape(batch[0][idx], [28,28])

# Okay better.
plt.imshow(img, interpolation="none")

# Hope to see a 4 here.
# Yay! It does indicate a 4.
batch[1][idx]

feed_dict={x: batch[0], y_ : batch[1]}

# What is this...? 
# A dictionary that has x,y as the key values,
# and the images (batch[0]) and class labels (batch[1])
type(feed_dict)

# Look at all the keys for this dictionary.
for key in feed_dict.keys():
    print(key)
    
# Hmmm... so the tensor is the key...

# Take 1 step.
train_step.run(feed_dict=feed_dict)

# Okay... so I think this takes ones gradient step.
# Now how can we check our predicted values?
# This gives an error...
y.eval()

tf.argmax(y,1).eval()



