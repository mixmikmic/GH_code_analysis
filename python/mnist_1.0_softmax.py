import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 28, 28, 1], 'training-data')

# create two variables
#  * W (a matrix) to hold the weights.
#  * b (a vector) to hold the biases.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Initialise the variables
init = tf.global_variables_initializer()

# Flatten the 28 x 28 training images into a single row of pixels.
XX = tf.reshape(X, [-1, 784])

# Define the model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# Placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10], 'correct-answers')

# Define the error function (use cross entropy)
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# Or alternatively
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0
# FIXME: What's the difference?

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# configure a GradientDescentOptimizer. A learning rate of `0.005` works well.
optimizer = tf.train.GradientDescentOptimizer(0.005)

train_step = optimizer.minimize(cross_entropy)

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

sess = tf.Session()
sess.run(init)

for i in range(1000):
    # Load a batch of 100 images and the correct answers for them
    batch_X, batch_Y = mnist.train.next_batch(100)
    
    train_data = {X: batch_X, Y_: batch_Y}

    # Run the training step (back propagation)
    sess.run(train_step, feed_dict=train_data)
    
    # Check the accuracy every 100 iterations of our training run
    if i % 100 == 0:
        # Accuracy against the *training* data
        #print(sess.run(accuracy, feed_dict=train_data))
        train_acc, train_ent = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        
        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        test_acc, test_ent = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        
        print("[TRAIN - acc:%1.2f ent: %10.4f]     [TEST -  - acc:%1.2f ent: %10.4f]" 
              % (train_acc, train_ent, test_acc, test_ent))
        
print('---------- FINAL ACCURACY ----------')
print(sess.run(accuracy, feed_dict=test_data))

