import os.path as op
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import tensorflow as tf

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

image, label = read_and_decode(op.expanduser(op.join('~', 'data_ucsf','cells_train.tfrecords')))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
im_1, lab_1 = sess.run([image, label])

im_1.shape

plt.imshow((im_1 * 255).astype(int), vmin=0, vmax=255)

lab_1

# get single examples
image, label = read_and_decode(op.expanduser(op.join('~', 'data_ucsf','cells_train.tfrecords')))

# The following groups the examples into batches randomly. This function requires explicitely setting the
# Shapes of the inputs, so we take advantage of the fact that we already pulled out one example above
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=40,
    capacity=400,
    shapes=(im_1.shape, lab_1.shape),
    min_after_dequeue=200)

# The model: y_pred = Wx + b
W = tf.Variable(tf.zeros([np.prod(im_1.shape), 3]))
b = tf.Variable(tf.zeros([3]))

y_pred = tf.matmul(tf.reshape(images_batch, [-1, np.prod(im_1.shape)]), W) + b
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, labels_batch)

# We can use this for monitoring:
loss_mean = tf.reduce_mean(loss)

# This is an optimizer that will be used for training:
train_op = tf.train.AdamOptimizer().minimize(loss)

# These variables are used for evaluation (helping to decide when to stop training):
image_eval, label_eval = read_and_decode(op.expanduser(op.join('~', 'data_ucsf', 'cells_eval.tfrecords')))

# We use a different batch of 40 every time: 
images_eval_batch, labels_eval_batch = tf.train.batch(
            [image_eval, label_eval], batch_size=40,
            shapes=(im_1.shape, lab_1.shape))

y_pred_eval = tf.matmul(tf.reshape(images_eval_batch, [-1, np.prod(im_1.shape)]), W) + b

correct_prediction_eval = tf.reduce_mean(
    tf.cast(
        tf.equal(
            tf.cast(
                tf.argmax(y_pred_eval, 1), tf.int32), labels_eval_batch), 
            tf.float32))

# These will be used for a final test:
image_test, label_test = read_and_decode(op.expanduser(op.join('~', 'data_ucsf', 'cells_test.tfrecords')))

# Take the whole set:
images_test_batch, labels_test_batch = tf.train.batch(
            [image_test, label_test], batch_size=169,
            shapes=(im_1.shape, lab_1.shape))

y_pred_test = tf.matmul(tf.reshape(images_test_batch, [-1, np.prod(im_1.shape)]), W) + b

correct_prediction_test = tf.reduce_mean(
    tf.cast(tf.equal(tf.cast(tf.argmax(y_pred_test, 1), tf.int32), labels_test_batch), tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

n_iterations = 0
mean_losses = []
mean_evals = []
max_iterations = 5000

while True:    
    # This is where learning actually takes place!
    _, loss_mean_val = sess.run([train_op, loss_mean])
    # Store the loss so we can look at it:
    mean_losses.append(loss_mean_val)
    # Every 10 learning iterations, we consider whether to stop:
    if np.mod(n_iterations, 10) == 0:
        mean_evals.append(sess.run(correct_prediction_eval))
        print("At step %s, mean evaluated accuracy is: %2.2f"%(n_iterations, mean_evals[-1]))
        # But we really only start thinking about stopping 
        # after 2000 iterations:
        if n_iterations > 2000:
            # Here's how we decide whether to keep going, 
            # based on the held-out data:            
            # If you are still improving, relative to recent past keep training:
            if mean_evals[-1] < (np.mean(mean_evals[-10:-1])):
                break

    # If we're still around iterate:
    n_iterations = n_iterations + 1  

    # If you kept going for very long, break anyway:
    if n_iterations > max_iterations:
        break

fig, ax = plt.subplots(1)
ax.plot(mean_losses)
ax2 = plt.twinx()
ax2.plot(np.arange(0, n_iterations+1, 10), mean_evals, 'ro')

p = sess.run(correct_prediction_test)

print(p)



