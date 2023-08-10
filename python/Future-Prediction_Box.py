import sys
sys.path.append('..')
from util import *
from util.parser import *
from util.img_kit import *
from util.notebook_display import *
from util.numeric_ops import *
from util.tf_ops import *

from IPython import display
import numpy as np
from scipy import ndimage
from scipy import misc
from os import walk
import os
import tensorflow as tf
from PIL import Image

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.figsize'] = (5.0, 5.0) # set default size of plots
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

train_collection = get_collection("../data/moving-box/future/train")
total_train = sum([x.shape[0] for x in train_collection])
print("\nAfter Augmentation: img_collections has {} collections, {} images in total".format(len(train_collection), total_train))

test_collection = get_collection("../data/moving-box/future/test")
total_test = sum([x.shape[0] for x in test_collection])
print("\nAfter Augmentation: img_collections has {} collections, {} images in total".format(len(test_collection), total_test))

def sample(collection, batch_size = 8, gap = 3):
    """
    Input:
        collection: [img_data] - list of ndarray
    Output:
        before_imgs, after_img - [batch size, 32, 32]
    """
    np.random.shuffle(collection)
    # get average number of training for each class
    n_collection = len(collection)
    num_per_collection = [x.shape[0] for x in collection]
    avg_num_per_class = 2 * int(np.ceil(batch_size/n_collection))
    # before-index for each class
    before_ind = []
    for i, imgs in enumerate(collection):
        try:
            s = np.random.choice(range(0, num_per_collection[i]- gap - 1), avg_num_per_class, replace=False)
            before_ind.append(s)
        except: # if not enough in this class
            before_ind.append(np.array([]))
    # after-index for each class
    after_ind = [x+gap+1 for x in before_ind]
    
    selected_classes = [i for i in range(n_collection) if before_ind[i].shape[0]>0]
    before_imgs = np.concatenate([collection[i][before_ind[i]] for i in selected_classes], axis = 0)
    after_imgs = np.concatenate([collection[i][after_ind[i]] for i in selected_classes], axis = 0)
    
    before_imgs = before_imgs[:batch_size]
    after_imgs = after_imgs[:batch_size]
    assert len(before_imgs) == batch_size
    assert len(after_imgs) == batch_size
    return before_imgs, after_imgs


def sample_train(batch_size = 8, gap=3): return sample(train_collection, batch_size, gap)

def sample_test(batch_size = 8, gap=3):  return sample(test_collection, batch_size, gap)

def show_sample_train(batch_size, gap = 3):
    before, after = sample_train(batch_size=batch_size, gap = gap)
    print(before.shape)
    print("Range of Image Piece Value: [{}, {}]".format(np.min(before), np.max(before)))
    print("Before: {}".format(before.shape))
    print("After:  {}".format(after.shape))
    size = (20, 3)
    plot_images(before, size, "Before")
    plot_images(after, size, "After")
    
show_sample_train(batch_size = 6, gap = 9)

gap = 3

feature_channel = 128

batch_size = 32

learning_rate = 8e-5

beta = 0.7 # defalut 0.9 for adam
num_iteration = 8000
# curr_time = time()
# model_save_path = "../trained_model/box-64x64/{}/".format(curr_time)
# curve_save_path = "../output/learning_curve/box-64x64-{}".format(curr_time)

def encode(img, is_training=True):
    """
    Input:
        batch size of img
    Output:
        batch size of feature [batch_size, 8, 8, feature_channel]
    """
    x = tf.reshape(img, [-1, 64, 64, 1])
    x = tf.layers.conv2d(x, filters = 32, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)
    
    x = tf.layers.conv2d(x, filters = 64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)

    x = tf.layers.conv2d(x, filters = 64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)

    x = tf.layers.conv2d(x, filters = 128, kernel_size=2, padding='same', activation=tf.nn.relu)
    
    x = tf.layers.conv2d(x, filters = feature_channel, kernel_size=2, padding='same', activation=tf.nn.relu)
    return x

def decode(feature1, feature2, is_training=True):
    """
    Input:
        batch size of feature [batch_size, 8, 8, feature_channel]
    Output:
        batch size of img [batch_size, 32, 32, 1]
    """
    x = tf.concat([feature1, feature2], axis=3)

    x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=2, strides=1, activation=tf.nn.relu, padding='same')
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same')
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3,  strides=1, activation=tf.nn.tanh, padding='same')
    x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=2,  strides=1, activation=tf.nn.tanh, padding='same')
    img = tf.layers.conv2d_transpose(x, filters=1, kernel_size=2,  strides=1, activation=tf.nn.tanh, padding='same')
    return img

