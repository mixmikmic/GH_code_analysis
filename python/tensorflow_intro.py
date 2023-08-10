# first we need some data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_samples, batch_labels = mnist.train.next_batch(100)

batch_samples.shape

batch_labels.shape

image = batch_samples[0]
image

image = image.reshape((28, 28))
from matplotlib.pyplot import imshow
get_ipython().magic('matplotlib inline')
imshow(image, cmap='gray')

label = batch_labels[0]
label

import tensorflow as tf

# define a symbolic variable for the image input
# None is special syntax in most deep learning frameworks,
# it means this dimension might vary (it's the batch size)
x = tf.placeholder(dtype=tf.float32, 
                   shape=[None, 784])

# define symbolic variables for the parameters of our function
# tf.Variables are like function parameters, they are not the 
# arguments of the function, they define what the function
W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))

# Now we will construct our function expression:
#
# Wx + b
#
# It's a linear transformation of the input
preds = tf.matmul(x, W) + b

# We now have transformed x to a 10-dim. vector,
# but we want to interpret each output as the probability
# of each of the 10 classes: and discrete probabilities sum up to 1.
# So we apply a softmax() to make this happen:
preds = tf.nn.softmax(preds)

# this is now an argument (input) of the cost function,
# and not a parameter - hence tf.placeholder
targets = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# now we define the cross_entropy formula
temp = - targets * tf.log(preds)

# but we need to sum over axis 1 (0 index),
# that's the 10-dim axis
temp = tf.reduce_sum(temp, reduction_indices=[1])

# and then take the mean over all samples in the batch
cross_entropy_loss = tf.reduce_mean(temp)

# we define our optimization method
# it will automatically do the derivatives for us
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

# and then we take the update step:
#
# params = params - learning_rate * gradient
#
# from it
# NOTE: this is still a symbolic !!
update_step = optimizer.minimize(loss=cross_entropy_loss,
                                var_list=[W, b])

# we need a session to actually run expressions
sess = tf.InteractiveSession()

# we initialize all global variables
# those are W and b in this case
tf.global_variables_initializer().run()

# and now we do 1000 optimizations
for _ in range(1000):
    batch_samples, batch_labels = mnist.train.next_batch(100)
    
    # we run the update_step from above, feeding in actual values
    # for x and targets
    sess.run(update_step, feed_dict={x: batch_samples,
                                    targets: batch_labels})

# we define a symbolic expression to tell if a prediction is
# correct or not

# we need the argmax because preds is between 0 and 1, not
# only 0 and 1
correct_prediction = tf.equal(tf.argmax(preds,1), tf.argmax(targets,1))

# accuracy is the percentage of correct predictions over all
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# and now we execute the above symbolic expression on the actual data
# NOTE: we use a test set to check the accuracy, which the model has not
# seen before
sess.run(accuracy, feed_dict={x: mnist.test.images, 
                            targets: mnist.test.labels})

