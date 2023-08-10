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
channels = 1
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

# Hidden layer size.
layer_size_1 = 32
layer_size_2 = 32

# NOTE: The name of the variable is optional.
x = tf.placeholder(tf.float32, shape=(None, 784), name="X")
y = tf.placeholder(tf.float32, shape=(None, 10), name="Y")
lr_rate = tf.placeholder(tf.float32, shape=(), name="lr")
input_layer = tf.reshape(x, [-1, img_h, img_w, channels])

# https://www.tensorflow.org/tutorials/layers
# Convolutional Layer #1
conv1 = tf.layers.conv2d(inputs=input_layer,
                         filters=32,
                         kernel_size=[5, 5],
                         padding="same",
                         activation=tf.nn.relu)
# Pooling layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2
conv2 = tf.layers.conv2d(inputs=pool1,
                         filters=32,
                         kernel_size=[5, 5],
                         padding="same",
                         activation=tf.nn.relu)
# Pooling layer #2
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])

# Weight & bias.
# Hidden layer.
w_1 = tf.get_variable(shape=[7 * 7 * 32, layer_size_1], name="w_1",
                      initializer=tf.random_normal_initializer())
b_1 = tf.get_variable(shape=[layer_size_1], name="b_1",
                      initializer=tf.random_normal_initializer())

# Output layer.
w_o = tf.get_variable(shape=[layer_size_1, 10], name="w_o",
                      initializer=tf.random_normal_initializer())
b_o = tf.get_variable(shape=[10], name="b_o",
                      initializer=tf.random_normal_initializer())

# NOTE: Initializations are important.
# Zero initialization: initializer=tf.zeros_initializer())

# Compute predicted Y.
h_1 = tf.nn.relu(tf.add(tf.matmul(pool2_flat, w_1), b_1)) # <--- Add ReLU activation.
# h_1 = tf.sigmoid(tf.add(tf.matmul(pool2_flat, w_1), b_1)) # <--- Add Sigmoid activation.
y_pred = tf.nn.softmax(tf.add(tf.matmul(h_1, w_o), b_o))

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y, tf.log(y_pred)), axis=1))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y,
                                                          tf.log(tf.clip_by_value(y_pred,
                                                                                  1e-10,1.0))),
                                                          axis=1))

# The tensorflow function available.
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
#                                                                        logits=y_pred))

# Create a gradient descent optimizer with the set learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_rate)

# Run the optimizer to minimize loss
# Tensorflow automatically computes the gradients for the loss function!!!
train = optimizer.minimize(cross_entropy)

# Gradient Clipping.
# gvs = optimizer.compute_gradients(cross_entropy)
# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
# train = optimizer.apply_gradients(capped_gvs)

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
variable_summaries(w_1, "weights")
variable_summaries(b_1, "bias")
variable_summaries(cross_entropy, "loss")

# Initialize all variables
init = tf.global_variables_initializer()

# First create the correct prediction by taking the maximum value from the prediction class
# and checking it with the actual class. The result is a boolean column vector
correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Define some hyper-parameters.
lr = 0.01
epochs = 5
batch_size = 55
log_dir = 'logs/lenet/tf/' # Tensorboard log directory.
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
            y_p, curr_w, curr_b,            curr_loss, _, summary, cur_acc = sess.run([y_pred, w_1, b_1, cross_entropy,
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
                print("Epoch: %d/%d. Batch #: %d/%d. Loss: %.2f. Train Accuracy: %.2f"
                      %(epoch+1, epochs, batch_num, num_batches, curr_loss, cur_acc))
    
    # Testing.
    test_accuracy = sess.run(accuracy, feed_dict={x: test_x,                                                   y: test_y})
    print("Test Accuracy: %.2f"%test_accuracy)
    train_writer.close() # <-------Important!

from keras.layers import Dense, Input, Conv2D, Reshape, MaxPooling2D, Flatten
from keras.initializers import random_normal
from keras.models import Model
from keras import optimizers, metrics

from keras.callbacks import TensorBoard

# Create a layer to take an input.
input_l = Input(shape=np.array([784]))
input_r = Reshape((28, 28, 1))(input_l)

# Add convolutional layer-1.
conv1 = Conv2D(32, (5, 5), padding='same', activation='relu')(input_r)
max1 = MaxPooling2D(pool_size=(2, 2))(conv1)

# Add convolutional layer-2.
conv2 = Conv2D(32, (5, 5), padding='same', activation='relu')(max1)
max1 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat = Flatten()(max1)

# Compute Wx + b.
dense_1 = Dense(layer_size_1, activation='relu')(flat) # <-- Thats it!
output = Dense(10, activation='softmax')(dense_1)

# Create a model and compile it.
model = Model(inputs=[input_l], outputs=[output])
model.summary() # Get the summary.

sgd = optimizers.sgd(lr=lr)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# NOTE: Add Tensorboard after compiling.
tensorboard = TensorBoard(log_dir="logs/lenet/keras/")

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

