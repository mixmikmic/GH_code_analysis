get_ipython().run_line_magic('matplotlib', 'inline')
import sys
sys.path.insert(0, '../')
from skimage.io import imread
import tensorflow as tf
import utils as utils
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.transform import resize

## For bug fix
import importlib
importlib.reload(utils)

PATH_TO_HR = '../test_data/high_resolution/'
PATH_TO_LR = '../test_data/low_resolution/'
SAVE_DIR = './test_save'
NAME = 'valid_data'

test_data_obj = utils.Data(PATH_TO_LR, PATH_TO_HR, SAVE_DIR, NAME)
input_list, output_list = test_data_obj.get_files()
test_data_obj.convert_to_tfrecord(input_list, output_list)

filename = tf.constant(os.path.join(SAVE_DIR, NAME+'.tfrecords'))
dataset = tf.contrib.data.TFRecordDataset(filename)

def _parse_function(example_proto):
    features = {'in_shape': tf.FixedLenFeature([], tf.string),
                'out_shape': tf.FixedLenFeature([], tf.string),
                'in_image_raw': tf.FixedLenFeature([], tf.string),
                'out_image_raw': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    in_shape = tf.decode_raw(parsed_features["in_shape"], tf.int32)
    in_image = tf.decode_raw(parsed_features["in_image_raw"], tf.int32)
    out_shape = tf.decode_raw(parsed_features["out_shape"], tf.int32)
    out_image = tf.decode_raw(parsed_features["out_image_raw"], tf.int32)
    
    in_image = tf.reshape(in_image, in_shape)
    out_image = tf.reshape(out_image, out_shape)
    
    return in_image, out_image

dataset = dataset.map(_parse_function)
batched_dataset = dataset.batch(1)          # Since the images has different shapes, the batch set to 1.
iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()
sess = tf.InteractiveSession()
out = sess.run(next_element)
image_tf = out[0].squeeze()

image_1 = imread('../test_data/low_resolution/0001.png')
plt.figure(figsize=(20, 40))
plt.imshow(np.concatenate([image_1, image_tf], axis=1), cmap='gray')

CROP_SIZE_IN = 64
CROP_SIZE_OUT = CROP_SIZE_IN * 4
CHANNEL_IN = 1
def _preprocessing_function(image_in, image_out):
    
    in_shape = tf.shape(image_in)
    h_offset = tf.random_uniform([], minval=0, maxval=in_shape[0] - CROP_SIZE_IN, dtype=tf.int32)
    w_offset = tf.random_uniform([], minval=0, maxval=in_shape[1] - CROP_SIZE_IN, dtype=tf.int32)
    box_start = tf.stack([h_offset, w_offset, tf.constant(0)])
    box_size = tf.constant((CROP_SIZE_IN, CROP_SIZE_IN, CHANNEL_IN))
    cropped_image_in = tf.slice(image_in, box_start, box_size)

    out_shape = tf.shape(image_out)
    h_offset = tf.scalar_mul(4, h_offset)
    w_offset = tf.scalar_mul(4, w_offset)  
    box_start = tf.stack([h_offset, w_offset, tf.constant(0)])
    box_size = tf.constant((CROP_SIZE_OUT, CROP_SIZE_OUT, CHANNEL_IN))
    cropped_image_out = tf.slice(image_out, box_start, box_size)

    return cropped_image_in, cropped_image_out

dataset = tf.contrib.data.TFRecordDataset(filename)
dataset = dataset.map(_parse_function)
dataset = dataset.map(_preprocessing_function)
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(1)
dataset = dataset.repeat()
iterator = dataset.make_one_shot_iterator()
next_item = iterator.get_next()
sess = tf.InteractiveSession()
for _ in range(4):
    cropped_image_in, cropped_image_out = map(lambda x: x.squeeze(), sess.run(next_item))
    cropped_image_in = resize(cropped_image_in, cropped_image_out.shape)
    cropped_image_out = resize(cropped_image_out, cropped_image_out.shape)
    plt.figure(figsize=(10, 20))
    plt.imshow(np.concatenate([cropped_image_in, cropped_image_out], axis=1), cmap='gray')

