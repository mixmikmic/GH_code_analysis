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
import layers as layers

SHAPE_LOW = (64, 64)
SHAPE_HIGH = (256, 256)

list_images_low = np.array([np.ones(SHAPE_LOW)*scalar for scalar in np.linspace(0.0, 1.0, 4)]).reshape(SHAPE_LOW+(4,))

images_low = np.transpose(np.array([np.ones(SHAPE_LOW)*scalar for scalar in np.linspace(0.0, 0.9, 4)]), [1, 2, 0])
images_low = images_low.reshape((1,) + SHAPE_LOW + (4,))

plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.axis('off')
    plt.imshow(images_low[0, :, :, i], cmap='gray', vmin=0, vmax=1)

t_image = tf.constant(images_low, dtype=tf.float32)
x = layers.subpixel(t_image, 2, 'subpixel_shuffle_1')
sess = tf.InteractiveSession()
out = sess.run(x)
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(out.squeeze(), cmap='gray', vmin=0.0, vmax=1.0)

red = np.array([255, 0, 0]) / 255.
green = np.array([0, 255, 0]) / 255.
blue = np.array([0, 0, 255]) / 255.
yellow = np.array([255, 0, 255]) / 255.
scalars = list([red, green, blue, yellow])

images_low = [np.ones((64, 64, 1)) * scalar for scalar in scalars]
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.axis('off')
    plt.imshow(images_low[i])

images_low = np.concatenate(np.array(images_low), axis=2).reshape((1,) + SHAPE_LOW + (12,))
t_image = tf.constant(images_low, dtype=tf.float32)
x = layers.subpixel(t_image, 2, 'subpixel_shuffle_2')
out = sess.run(x)
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(out.squeeze(), cmap='gray', vmin=0.0, vmax=1.0)

