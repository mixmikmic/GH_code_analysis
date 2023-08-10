import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import tensorflow as tf
import os.path as op

log_dir = './logs'
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)

checkpoint_dir = './checkpoints'

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/depth': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
                'image/raw': tf.VarLenFeature(tf.string)})

    # Shape elements must be int32 tensors!
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    depth = tf.cast(features['image/depth'], tf.int32)
    
    # Decode the image from its raw representation:
    image = tf.decode_raw(features['image/raw'].values, tf.uint8)

    # Reshape it back to its original shape:
    im_shape = tf.pack([height, width, depth])
    image = tf.reshape(image, im_shape)
    #tf.random_crop(image, [height, width, depth])
    # Convert from [0, 255] -> [0, 1] floats.
    image = tf.cast(image, tf.float32) * (1. / 255)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    return image, label

image, label = read_and_decode(op.expanduser(op.join('~', 'data_ucsf', "cells_train.tfrecords")))
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
im_1, lab_1 = sess.run([image, label])

image, label = read_and_decode(op.expanduser(op.join('~', 'data_ucsf', "cells_train.tfrecords")))

with tf.name_scope('input'):
    images_batch, labels_batch = tf.train.shuffle_batch(
        [image, label], batch_size=40,
        capacity=400,
        shapes=(im_1.shape, lab_1.shape),
        min_after_dequeue=200)

def weight_variable(name, shape):
    """ 
    Initialize weights with the Xavier initialization
    """
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("First_convolutional_layer"):
    with tf.name_scope("Weights"):
        W_conv1 = weight_variable("W_conv1", [5, 5, 3, 32])
        variable_summaries(W_conv1)
    with tf.name_scope("Bias"):
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1)
    with tf.name_scope('Preactivation'):
        preact1 = conv2d(images_batch, W_conv1) + b_conv1
        tf.summary.histogram('preactivation', preact1)

    h_conv1 = tf.nn.relu(preact1)
    tf.summary.histogram("convolution", h_conv1)
    with tf.name_scope("Pooling"):
        h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope("Second_convolutional_layer"):
    with tf.name_scope("Weights"):
        W_conv2 = weight_variable("W_conv2", [5, 5, 32, 64])
        variable_summaries(W_conv2)
    with tf.name_scope("Bias"):
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2)
    with tf.name_scope("Preactivation"):
        preact2 = conv2d(h_pool1, W_conv2) + b_conv2
        tf.summary.histogram('preactivation', preact2)
    h_conv2 = tf.nn.relu(preact2)
    tf.summary.histogram("convolution", h_conv2)
    with tf.name_scope("Pooling"):
        h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope("Fully_connected_layer"):
    with tf.name_scope("Weights"):
        W_fc1 = weight_variable("W_fc1", [64 * 64 * 64, 1024])
        variable_summaries(W_fc1)
    with tf.name_scope("Bias"):
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1)
        
    h_pool2_flat = tf.reshape(h_pool2, [-1, 64 * 64 * 64])
    with tf.name_scope("Preactivation"):
        preact3 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        variable_summaries(preact3)
    
    h_fc1 = tf.nn.relu(preact3)
    tf.summary.histogram('Fully_connected_output', h_fc1)

with tf.name_scope("Dropout"):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('Keep_Probability', keep_prob)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope("Readout"):
    with tf.name_scope("Weights"):
        W_fc2 = weight_variable("W_fc2", [1024, 3])
        variable_summaries(W_fc2)
    with tf.name_scope("Bias"):
        b_fc2 = bias_variable([3])
        variable_summaries(b_fc2)
    with tf.name_scope("Activation"):
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        variable_summaries(y_conv)

with tf.name_scope("Loss"):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, labels_batch)
    loss_mean = tf.reduce_mean(loss)
    tf.summary.scalar('Mean_loss', loss_mean)
with tf.name_scope("Train"):
    train_op = tf.train.AdamOptimizer(1e-5).minimize(loss_mean)

y_pred = tf.cast(tf.argmax(y_conv, 1), tf.int32)

with tf.name_scope("Accuracy"):
    with tf.name_scope("Correct_prediction"):
        correct_prediction = tf.equal(y_pred, labels_batch)
    with tf.name_scope("Accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

with tf.name_scope("Evaluation"):

    # These variables are used for evaluation (helping to decide when to stop training):
    image_eval, label_eval = read_and_decode(op.expanduser(op.join('~', 'data_ucsf', "cells_eval.tfrecords")))

    # We use a different batch of 40 every time: 
    images_eval_batch, labels_eval_batch = tf.train.batch(
                [image_eval, label_eval], batch_size=40,
                shapes=(im_1.shape, lab_1.shape))

    # Reproducing the entire network on eval data:
    h_conv1_eval = tf.nn.relu(conv2d(images_eval_batch, W_conv1) + b_conv1)
    h_pool1_eval = max_pool_2x2(h_conv1_eval)

    h_conv2_eval = tf.nn.relu(conv2d(h_pool1_eval, W_conv2) + b_conv2)
    h_pool2_eval = max_pool_2x2(h_conv2_eval)

    h_pool2_flat_eval = tf.reshape(h_pool2_eval, [-1, 64 * 64 * 64])
    h_fc1_eval = tf.nn.relu(tf.matmul(h_pool2_flat_eval, W_fc1) + b_fc1)

    y_pred_eval = tf.matmul(h_fc1_eval, W_fc2) + b_fc2

    correct_prediction_eval = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.cast(
                    tf.argmax(y_pred_eval, 1), tf.int32), labels_eval_batch), 
                tf.float32))

    tf.summary.scalar('correct_prediction_eval', correct_prediction_eval)

# These will be used for a final test:
image_test, label_test = read_and_decode(op.expanduser(op.join('~', 'data_ucsf', "cells_test.tfrecords")))

# Use the whole thing  
images_test_batch, labels_test_batch = tf.train.batch(
            [image_test, label_test], batch_size=40,
            shapes=(im_1.shape, lab_1.shape))

# Reproducing the entire network on eval data:
h_conv1_test = tf.nn.relu(conv2d(images_test_batch, W_conv1) + b_conv1)
h_pool1_test = max_pool_2x2(h_conv1_test)

h_conv2_test = tf.nn.relu(conv2d(h_pool1_test, W_conv2) + b_conv2)
h_pool2_test = max_pool_2x2(h_conv2_test)

h_pool2_flat_test = tf.reshape(h_pool2_test, [-1, 64 * 64 * 64])
h_fc1_test = tf.nn.relu(tf.matmul(h_pool2_flat_test, W_fc1) + b_fc1)

y_pred_test = tf.matmul(h_fc1_test, W_fc2) + b_fc2

correct_prediction_test = tf.reduce_mean(
    tf.cast(
        tf.equal(
            tf.cast(
                tf.argmax(y_pred_test, 1), tf.int32), labels_test_batch), 
            tf.float32))

sess = tf.Session()

merged = tf.summary.merge_all()
log_dir = './logs'
train_writer = tf.summary.FileWriter(op.join(log_dir, 'train'), sess.graph)
eval_writer = tf.summary.FileWriter(op.join(log_dir, 'evaluation'))

init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

n_iterations = 0
mean_losses = []
mean_evals = []
max_iterations = 5000

while True:    
    _, loss_mean_val, summary = sess.run([train_op, loss_mean, merged], feed_dict={keep_prob: 0.5})
    mean_losses.append(loss_mean_val)
    # Write summary into the training writer
    train_writer.add_summary(summary, n_iterations)
    # Every 10 learning iterations, we consider whether to stop:
    if np.mod(n_iterations, 10) == 0:
        p, summary = sess.run([correct_prediction_eval, merged], feed_dict={keep_prob: 1.0})
        mean_evals.append(p)
        print("At step %s, mean evaluated accuracy is: %2.2f"%(n_iterations, mean_evals[-1]))
        # We've taken out the breaking criterion!
        eval_writer.add_summary(summary, n_iterations)

    n_iterations = n_iterations + 1  

    # If you kept going for very long, break anyway:
    if n_iterations > max_iterations:
        break

train_writer.close()
eval_writer.close()

p = sess.run(correct_prediction_test)

print(p)

# saver = tf.train.Saver()
# saver.save(sess, op.join(checkpoint_dir, "model.ckpt"), i)

from tensorflow.contrib.tensorboard.plugins import projector
summary_writer = tf.summary.FileWriter(log_dir)

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = "W_conv1"
# Link this tensor to its metadata file (e.g. labels).
#embedding.metadata_path = op.join(log_dir, 'metadata.tsv')

# Saves a configuration file that TensorBoard will read during startup.
projector.visualize_embeddings(summary_writer, config)



