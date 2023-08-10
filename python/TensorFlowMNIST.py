import tensorflow.examples.tutorials.mnist.input_data as id
mnist = id.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
#inputs and outputs
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# weights and biases are defined, all initialized to zero
sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(x, W) + b)
# activation function (softmax as opposed to logistic)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# this isn't the cross-entropy function I saw in Nielsen's overview, but seems to be standard.
# perhaps try this with nielsen's as well
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# automatically computes derivatives
# learning rate is 0.01
epochs = 1000
for i in range(epochs):
    batch = mnist.train.next_batch(50) # training with batches of 50 from the training data
    train_step.run(feed_dict={x: batch[0], y_:batch[1]}) # placeholders must always be fed
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # argmax returns the index of the largest value, i.e. the predicted number
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# this is the accuracy on the test data after training
# eval on a tensor is the same as passing the tensor to sess.run

# i.e. no negatives
image_height = 28
image_width = 28
conv1_filters = 50
conv2_filters = 70
filter_size = 5
output_length = 10
batch_size = 50
fully_connected_length = int((image_height + filter_size - 1) * (image_width + filter_size - 1))
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# all bias variables initialized to 0.1
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# this is a convolution layer
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([filter_size, filter_size, 1, conv1_filters])
b_conv1 = bias_variable([conv1_filters])
x_image = tf.reshape(x, [-1,image_height,image_width,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_fc1 = weight_variable([int(image_height/4 * image_width/4 * conv2_filters), 
                         fully_connected_length])
b_fc1 = bias_variable([fully_connected_length])
W_conv2 = weight_variable([filter_size, filter_size, conv1_filters, conv2_filters])
b_conv2 = bias_variable([conv2_filters])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, int(image_height/4*image_width/4*conv2_filters)])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([fully_connected_length, output_length])
b_fc2 = bias_variable([output_length])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(1001):
  batch = mnist.train.next_batch(batch_size)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



