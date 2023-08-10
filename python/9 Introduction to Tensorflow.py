from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf

sess = tf.InteractiveSession()

# Python is great for ease of use.  But it's not fast.
# C++ is great for speed, but it is not easy to use.
# Tensor flow lets us specify what we want to do in Python, and then do things in C++.

# Placeholders are containers for data.  We set them up with the expectation that we will
# be 'feeding' them something.

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# What is being fed into x?
# What is being fed into y?

# Variables are containers for numbers.  We specify them here specifically
# so that when tensorflow talks to C++ there is no confusion.

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# What is the session doing here?

sess.run(tf.initialize_all_variables())

# Set up regression model

y = tf.matmul(x,W) + b

# Softmax is just logistic regression

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# What is train_step?

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train the model

for i in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Evaluate the model.  How well did it do?

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



