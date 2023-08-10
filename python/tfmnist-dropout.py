# Importing Tensorflow
import tensorflow as tf
from tensorflow.python.framework import ops

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

# Let's first reset our graph, so our neural network components are all declared within the same graph
ops.reset_default_graph() 

X = tf.placeholder(tf.float32, [784, None])
Y = tf.placeholder(tf.float32, [10, None])
D = tf.placeholder(tf.float32, [])
L = tf.placeholder(tf.float32, [])

tf.set_random_seed(1)   # Set the same seed to get the same initialization as in this demo.

# The weights for the input layer
W1 = tf.get_variable("W1", [64, 784], initializer=tf.contrib.layers.xavier_initializer(seed=1))

# The bias for the output from the input layer
b1 = tf.get_variable("b1", [64, 1], initializer=tf.zeros_initializer())

# The first layer (input layer)
Z1 = tf.add(tf.matmul(W1, X), b1)

# Let's add the activation function to the output signal from the first layer
A1 = tf.nn.relu(Z1)

W2 = tf.get_variable("W2", [32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b2 = tf.get_variable("b2", [32, 1], initializer=tf.zeros_initializer())

# The second layer (first hidden layer)
Z2 = tf.add(tf.matmul(W2, A1), b2) 

# Let's add the dropout layer to the output signal from the second layer
D2 = tf.nn.dropout(Z2, keep_prob=D)

# Let's add the activation function to the output signal from the dropout layer
A2 = tf.nn.relu(D2)

W3 = tf.get_variable("W3", [20, 32], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b3 = tf.get_variable("b3", [20, 1], initializer=tf.zeros_initializer())

# The third layer (second hidden layer)
Z3 = tf.add(tf.matmul(W3, A2), b3) 

# Let's add the activation function to the output signal from the third layer
A3 = tf.nn.relu(Z3)

W4 = tf.get_variable("W4", [10, 20], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b4 = tf.get_variable("b4", [10, 1], initializer=tf.zeros_initializer())

# The fourth layer (output layer)
Z4 = tf.add(tf.matmul(W4, A3), b4) 

# to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.transpose(Z4), labels=tf.transpose(Y)))

# The learning rate for Gradient Descent algorithm
#learning_rate = 0.5

optimizer = tf.train.GradientDescentOptimizer(L).minimize(cost)

init = tf.global_variables_initializer()

import time

epochs = 20                                    # run a 20 epochs
batch_size = 200                               # for each epoch, train in batches of 200 images
number_of_images = mnist.train.labels.shape[0] # number of images in training data
batches = number_of_images // batch_size       # number of batches in an epoch
print("Number of batches:", batches)

keep_prob = 0.9                                # percent of outputs to keep in dropout layer
learning_rate = 0.5                            # the learning rate for graident descent

start = time.time()

with tf.Session() as sess:
    # Initialize the variables
    sess.run(init)
    
    # run our training data through the neural network for each epoch
    for epoch in range(epochs):
        
      epoch_cost = 0
      
      # Run the training data through the neural network
      for batch in range(batches):
          # Get a batch (random shuffled) from the training data
          batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      
          # Feed this batch through the neural network.
          _, batch_cost = sess.run([optimizer, cost], feed_dict={X: batch_xs.T, Y: batch_ys.T, D: keep_prob, L: learning_rate})
            
          epoch_cost += batch_cost
      
      print("Epoch: ", epoch, epoch_cost / batches)
        
    end = time.time()
    
    print("Training Time:", end - start)
    
    # Test the Model
    
    # Let's select the highest percent from the softmax output per image as the prediction.
    prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))
    
    # Let's create another node for calculating the accuracy
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    # Now let's run our trainingt images through the model to calculate our accuracy during training
    # Note how we set the keep percent for the dropout rate to 1.0 (no dropout) when we are evaluating the accuracy.
    print ("Train Accuracy:", accuracy.eval({X: mnist.train.images.T, Y: mnist.train.labels.T, D: 1.0}))
    
    # Now let's run our test images through the model to calculate our accuracy on the test data
    print ("Test Accuracy:", accuracy.eval({X: mnist.test.images.T, Y: mnist.test.labels.T, D: 1.0}))

# Let's drop out 75% of our nodes
learning_rate = 0.05

# Let's drop out 50% of our nodes
keep_prob = 0.50

keep_prob = 0.90
learning_rate = 0.05

learning_rate = 0.9



