import time
from IPython import display

# Import the libraries and load the datasets.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# import MNIST data.
from tensorflow.examples.tutorials.mnist import input_data

# Check previous section for details on MNIST dataset.
mnist = input_data.read_data_sets("data/", one_hot=True)

# Define some standard parameters.
img_h = 28
img_w = 28
n_classes = 10

# Training, validation, testing...
train_x = mnist[0].images
train_y = mnist[0].labels
print("Training Size: {}".format(len(train_x)))

val_x = mnist[1].images
val_y = mnist[1].labels
print("Validation Size: {}".format(len(val_x)))

test_x = mnist[2].images
test_y = mnist[2].labels
print("Test Size: {}".format(len(test_x)))

# NOTE: The name of the variable is optional.
x = tf.placeholder(tf.float32, shape=(None, 784), name="X")
y = tf.placeholder(tf.float32, shape=(None, 10), name="Y")
lr_rate = tf.placeholder(tf.float32, shape=(), name="lr")

# Weight & bias.
w = tf.get_variable(shape=[784, 10], name="w", initializer=tf.zeros_initializer())
b = tf.get_variable(shape=[10], name="b", initializer=tf.zeros_initializer())

# Compute predicted Y.
y_pred = tf.nn.softmax(tf.add(tf.matmul(x, w), b))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y, tf.log(y_pred)), axis=1))

# The tensorflow function available. Use tf.reduce_mean for a batch.
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
#                                                                        logits=y_pred))

# Create a gradient descent optimizer with the set learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_rate)

# Run the optimizer to minimize loss
# Tensorflow automatically computes the gradients for the loss function!!!
train = optimizer.minimize(cross_entropy)

# Helper function.
# https://www.tensorflow.org/get_started/summaries_and_tensorboard
def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    
# Define summaries.
variable_summaries(w, "weights")
variable_summaries(b, "bias")
variable_summaries(cross_entropy, "loss")

# Initialize all variables
init = tf.global_variables_initializer()

# First create the correct prediction by taking the maximum value from the prediction class
# and checking it with the actual class. The result is a boolean column vector
correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Define some hyper-parameters.
lr = 0.005
epochs = 5
batch_size = 55
log_dir = 'logs/logistic/tf/' # Tensorboard log directory.
batch_limit = 100

# Train the model.
with tf.Session() as sess:
    # Initialize all variables
    sess.run(init)
    
    # Create the writer.
    # Merge all the summaries and write them.
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    
    num_batches = int(len(train_x)/batch_size)
    for epoch in range(epochs):
        for batch_num in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            y_p, curr_w, curr_b,            curr_loss, _, summary, cur_acc = sess.run([y_pred, w, b, cross_entropy,
                                                      train, merged, accuracy],
                                                      feed_dict = {x: batch_xs,
                                                                   y: batch_ys,
                                                                   lr_rate: lr})
            if batch_num % batch_limit == 0:
                # IMP: Add the summary for each epoch.
                train_writer.add_summary(summary, epoch)
                display.clear_output(wait=True)
                time.sleep(0.1)

                # Print the loss
                print("Epoch: %d/%d. Batch #: %d/%d. Current loss: %.5f. Train Accuracy: %.2f"
                      %(epoch+1, epochs, batch_num, num_batches, curr_loss, cur_acc))
    
    # Test Accuracy.
    test_acc = sess.run([accuracy], feed_dict={x: test_x, y: test_y})
    print("Test Accuracy: {}".format(test_acc[0]))
    
    train_writer.close() # <-------Important!

from keras.layers import Dense, Input
from keras.initializers import random_normal
from keras.models import Model
from keras import optimizers, metrics

from keras.callbacks import TensorBoard

# Create a layer to take an input.
input_l = Input(shape=np.array([784]))
# Compute Wx + b.
dense = Dense(np.array([10]), activation='softmax')
output = dense(input_l)

# Create a model and compile it.
model = Model(inputs=[input_l], outputs=[output])
model.summary() # Get the summary.

sgd = optimizers.sgd(lr=lr)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# NOTE: Add Tensorboard after compiling.
tensorboard = TensorBoard(log_dir="logs/logistic/keras/")

# Train the model.
# Add a callback.
model.fit(x=train_x, y=train_y, batch_size=batch_size, 
          epochs=epochs, verbose=0, callbacks=[tensorboard])

# Predict the y's.
y_p = model.predict(test_x)
y_p_loss = model.evaluate(test_x, test_y)

# Plot them.
print("Evaluation Metrics: " + str(model.metrics_names))
print("Loss: {}, Accuracy: {}".format(y_p_loss[0], y_p_loss[1]))

