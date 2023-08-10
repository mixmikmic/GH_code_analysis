# Location of checkpoint directory from training
CHECKPOINT_DIR = "/tmp/mnist-inception"

# Number of classes (MNIST = 10, 1 for each digit)
NUM_CLASSES = 10

# Number of test images to evaluate at a time. Adjust to smaller if you're
# evaluation on a memory-constrained box
BATCH_SIZE = 20

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import inception_model as inception

from matplotlib import pyplot as plt, rcParams
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage.interpolation import zoom
from tensorflow.examples.tutorials.mnist import input_data

rcParams['figure.figsize'] = 20, 15

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

MEAN = np.mean(mnist.train.images)
STD = np.std(mnist.train.images)
NUM_TRAIN = mnist.train.labels.shape[0]
NUM_TEST = mnist.test.labels.shape[0]

print("Found %d training examples, %d test examples" % (NUM_TRAIN, NUM_TEST))

# A convenience method of plotting MNIST images in a grid
def plot_img(img, title, grid, index):
    grid[index].imshow(img)
    grid[index].set_title(title, fontsize=8)
    grid[index].axes.get_xaxis().set_visible(False)
    grid[index].axes.get_yaxis().set_visible(False)

# A convenience method for resizing the 784x1 monochrome images into
# the 299x299x3 RGB images that the Inception model accepts as input
RESIZE_FACTOR = (299/28)
def resize_images(images, mean=MEAN, std=STD):
    reshaped = (images - mean)/std
    reshaped = np.reshape(reshaped, [-1, 28, 28, 1]) # Reshape to 28x28x1

    # Reshape to 299x299 images, then duplicate the single monochrome channel
    # across 3 RGB layers
    zoomed = zoom(reshaped, [1.0, RESIZE_FACTOR, RESIZE_FACTOR, 1.0])
    zoomed = np.repeat(zoomed, 3, 3)

    return zoomed

def build_inference(sess, images, labels, checkpoint_dir):

    logits, _ = inception.inference(images, NUM_CLASSES + 1)

    variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    print("Restored at step = %s" % (global_step))
    
    return logits

sess = tf.InteractiveSession()
images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
labels = tf.placeholder(tf.float32, shape=(None, 1))

logits = build_inference(sess, images, labels, CHECKPOINT_DIR)

num_batches = NUM_TEST // BATCH_SIZE
bad_inds = []
bad_predictions = []

total_evaluated = 0
total_correct = 0
for index in xrange(num_batches):
    time_start = time.time()
    
    # Fetch the next batch, resize images and reshape labels to the appropriate shape
    raw_images, raw_labels = mnist.test.next_batch(BATCH_SIZE)
    batch_images = resize_images(raw_images)
    batch_labels = np.reshape(np.argmax(raw_labels, 1), (BATCH_SIZE, 1))
        
    # Use the Inception model to predict labels for this batch of test data
    result = sess.run(logits, feed_dict={ images: batch_images, labels: batch_labels })
    predicted_labels = np.argmax(result[:, 1:11], 1).reshape((BATCH_SIZE, 1))
    
    # Determine which predictsion were correct and keep a running tally of accuracy
    inds_incorrect = np.where(predicted_labels != batch_labels)
    n_correct = np.sum(predicted_labels == batch_labels)
    total_correct += n_correct
    total_evaluated += BATCH_SIZE
    accuracy = 100 * total_correct / total_evaluated
    
    # Save which test cases were predicted incorrectly for later visualization
    bad_inds.extend((index * BATCH_SIZE) + inds_incorrect[0])
    bad_predictions.extend(predicted_labels[inds_incorrect])
    
    if ((index % 50 == 0) or (index == (num_batches - 1))):
        elapsed = time.time() - time_start
        print("%3d: Elapsed: %4.1fs, Batch Correct = %2d, Total Correct = %5d, Total Evaluated = %5d, Acc = %7.3f%%" % (index, elapsed, n_correct, total_correct, total_evaluated, accuracy))

num_bad = len(bad_inds)
grid_width = 10
grid_height = (num_bad // grid_width) + 1
plt.set_cmap(plt.gray())
fig = plt.figure(1)
grid = ImageGrid(fig, 111, nrows_ncols=(grid_height, grid_width), axes_pad=0.4)

sorted_bad_inds = sorted(range(num_bad), key=lambda i: bad_predictions[i])

for i in xrange(num_bad):
    img_idx = bad_inds[sorted_bad_inds[i]]
    img = mnist.test.images[img_idx, :].reshape((28, 28))
    predicted_label = bad_predictions[sorted_bad_inds[i]]
    actual_label = np.argmax(mnist.test.labels[img_idx, :])
    
    label = "%d (Actual: %d)" % (predicted_label, actual_label)
    
    plot_img(img[:, :], label, grid, i)

plt.show()



