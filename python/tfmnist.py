# Importing Tensorflow
import tensorflow as tf

# Import the MNIST input_data function from the tutorials.mnist package
from tensorflow.examples.tutorials.mnist import input_data

# Read in the data
# The paramter one_hot=True refers to the label which is categorical (1-10). 
# The paramter causes the label to be re-encoded as a 10 column vector.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

type(mnist)

print( type(mnist.train) )
print( type(mnist.test) )
print( type(mnist.validation) )

type(mnist.train.images)

# Let's get the length (number of images) of the list of images.
print( len(mnist.train.images) )
print( len(mnist.train.labels) )

mnist.train.images[0]

import matplotlib.pyplot as plt

# This line is specific to python notebooks (not python). 
# It causes plots to automatically be rendered (displayed) without issuing a show command.
get_ipython().run_line_magic('matplotlib', 'inline')

# Let's show that the shape of the image is already flatten (will output as 784)
print( mnist.train.images[1].shape )

# Let's now reshape it into a 28 x 28 matrix
image = mnist.train.images[1].reshape(28,28)
print( image.shape )

# Let's plot it now
plt.imshow( image )

# As you can see, there is a 1 at index 3 (fourth location, starting at 0)
mnist.train.labels[1]

x = tf.placeholder(tf.float32, [None, 784])

# The weights for the input layer
W = tf.Variable(tf.zeros([784, 10]))

# The bias for the output from the input layer
b = tf.Variable(tf.zeros([10]))

input_layer = tf.matmul(x, W) + b

# Add the softmax activation function to the input layer
y = tf.nn.softmax(input_layer)

y_ = tf.placeholder(tf.float32, [None, 10])

# The cost function which accumulates the loss for a batch
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))

# The learning rate for Gradient Descent algorithm
learning_rate = 0.5

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

import time

epochs = 1000      # run a 1000 epochs
batch_size = 100   # for each epoch, train in batches of 100 images

start = time.time()

with tf.Session() as sess:
    # Initialize the variables
    sess.run(init)
    
    # run our training data through the neural network for each epoch
    for epoch in range(epochs):
      # Get a batch (random shuffled) from the training data
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      
      # Feed this batch through the neural network.
      sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})
      
      if epoch % 100 == 0:
        print("Epoch: ", epoch)
        
    end = time.time()
    
    print("Training Time:", end - start)
    
    # Test the Model
    
    # Let's select the highest percent from the softmax output per image as the prediction.
    prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    
    # Let's create another node for calculating the accuracy
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    
    # Now let's run our test images through the model to calculate our accuracy
    print("Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



