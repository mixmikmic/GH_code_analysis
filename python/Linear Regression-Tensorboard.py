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

# Plots.
import matplotlib.pyplot as plt

# Generate some random data.
n = 250 # Number of datapoints.
d = 1 # The dimensions of the datapoints.
slope = 12
bias = 10

# Randomly generate input.
sample_x = np.random.rand(n, d)

# Consider the equation of the y = m.x + b
sample_y = slope*sample_x + bias

# NOTE: We do not want a straight line. So add some random noise to the input.
sample_noise = np.random.rand(n, d)
sample_x += sample_noise

# Plot.
plt.scatter(sample_x, sample_y, marker='x')
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.show()

# NOTE: The name of the variable is optional.
x = tf.placeholder(tf.float32, shape=(None, 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")
lr_rate = tf.placeholder(tf.float32, shape=(), name="lr")

# Weight & bias.
# w = tf.get_variable(shape=[1], name="w", initializer=tf.zeros_initializer())
# b = tf.get_variable(shape=[1], name="b", initializer=tf.zeros_initializer())

# Initialize with a different value.
w = tf.Variable(np.array([[5.0]]), dtype=tf.float32, name="w")
b = tf.Variable(np.array([[5.0]]), dtype=tf.float32, name="b")

with tf.name_scope("weights"):
    w_temp = tf.Variable(np.array([[5.0]]), dtype=tf.float32, name="w")
    b_temp = tf.Variable(np.array([[5.0]]), dtype=tf.float32, name="b")
    

# Note the difference in how the names are!
print("Without Scope: {}".format(w.name))
print("Without Scope: {}".format(w_temp.name))

# Compute predicted Y.
y_pred = w*x + b

loss = tf.div(tf.reduce_mean(tf.square(y - y_pred)), 2*n)

# This can be done in a single line.
# loss = tf.losses.mean_squared_error(y, y_pred)

# Create a gradient descent optimizer with the set learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_rate)

# Run the optimizer to minimize loss
# Tensorflow automatically computes the gradients for the loss function!!!
train = optimizer.minimize(loss)

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
variable_summaries(loss, "loss")

# Initialize all variables
init = tf.global_variables_initializer()

# Define some hyper-parameters.
lr = 0.1
epochs = 5000
log_dir = 'logs/linear/tf/' # Tensorboard log directory.

# Train the model.
with tf.Session() as sess:
    # Initialize all variables
    sess.run(init)
    
    # Create the writer.
    # Merge all the summaries and write them.
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    
    for epoch in range(epochs):
        y_p, curr_w, curr_b, curr_loss, _, summary = sess.run([y_pred, w, b, loss,
                                                               train, merged],
                                                      feed_dict = {x:sample_x, y: sample_y,
                                                                  lr_rate: lr})
        # IMP: Add the summary for each epoch.
        train_writer.add_summary(summary, epoch)    
    
    train_writer.close() # <-------Important!
    print("W: {}, B: {}, Loss: {}".format(curr_w[0][0], curr_b[0][0], curr_loss))
    print("Slope: {}, Bias: {}".format(slope, bias))
    plt.scatter(sample_x, sample_y, marker='x')
    plt.scatter(sample_x, y_p, c='red', marker='o')
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.show()

from keras.layers import Dense, Input
from keras.initializers import random_normal
from keras.models import Model
from keras import optimizers, metrics

from keras.callbacks import TensorBoard

# Create a layer to take an input.
input_l = Input(shape=np.array([1]))
# Compute Wx + b.
dense = Dense(np.array([1]), activation='linear')
output = dense(input_l)

# Create a model and compile it.
model = Model(inputs=[input_l], outputs=[output])
model.summary() # Get the summary.

sgd = optimizers.sgd(lr=lr)
model.compile(optimizer=sgd, loss='mean_squared_error')

# NOTE: Add Tensorboard after compiling.
tensorboard = TensorBoard(log_dir="logs/linear/keras/")

# Train the model.
# Add a callback.
model.fit(x=sample_x, y=sample_y, epochs=epochs, verbose=0, callbacks=[tensorboard])

# Predict the y's.
y_p = model.predict(sample_x)
y_p_loss = model.evaluate(sample_x, sample_y)

# Plot them.
print("Evaluation Metrics: " + str(model.metrics_names))
print("W: {}, B: {}, Loss: {}".format(dense.get_weights()[0][0][0],
                                      dense.get_weights()[1][0], y_p_loss))
print("Slope: {}, Bias: {}".format(slope, bias))
plt.scatter(sample_x, sample_y, marker='x')
plt.scatter(sample_x, y_p, c='red', marker='o')
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.show()

