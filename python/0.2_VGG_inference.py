import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names

import matplotlib.pyplot as plt
from pprint import pprint

get_ipython().magic('matplotlib inline')

# dir for tensorboard file (for graph visualization)
logs_path = 'logs/0.2/'

if tf.gfile.Exists(logs_path):
    tf.gfile.DeleteRecursively(logs_path)
tf.gfile.MakeDirs(logs_path)

# information about image size
IMG_W = 224
IMG_H = 224
CHANNELS = 3

img1 = imread('images/dog.jpg')
img1 = imresize(img1, (IMG_W, IMG_H))
img1 = img1.reshape((1, IMG_W, IMG_H, CHANNELS))
print(img1.dtype)
plt.imshow(img1[0])

config = tf.ConfigProto()
# this type of configuration is for GPU 
# TensorFlow will not allow all the GPU memory 
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

VGG19_weights_file = "data/VGG/vgg19.npy"
VGG16_weights_file = "data/VGG/vgg16.npy"

from VGG import generate_VGG19, generate_VGG16

with tf.name_scope('VGG16_a'):
    vgg16_a, vgg16_scope = generate_VGG16(VGG16_weights_file,
                                          scope="VGG16_factory",
                                          remove_top=False,
                                          input_shape=(1, IMG_W, IMG_H, CHANNELS),
                                          input_tensor=None,
                                          apply_preprocess=True)

print(type(vgg16_a))
pprint(vgg16_a)
print(type(vgg16_scope))
print(vgg16_scope.name)

sess.run(tf.global_variables_initializer())
_ = sess.run(vgg16_a['input'].assign(img1))

prob = sess.run(vgg16_a['prob'])[0]
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])

writer = tf.summary.FileWriter(logs_path, sess.graph)

with tf.name_scope('VGG16_b'):
    vgg16_b, _ = generate_VGG16(VGG16_weights_file,
                                scope=vgg16_scope,
                                remove_top=False,
                                input_shape=(1, IMG_W, IMG_H, CHANNELS),
                                input_tensor=None,
                                apply_preprocess=True)

print(type(vgg16_b))
print(type(vgg16_scope))

# need to perform again the variable initialization for vgg16_b
sess.run(tf.global_variables_initializer())
_ = sess.run(vgg16_a['input'].assign(img1))
_ = sess.run(vgg16_b['input'].assign(img1))

prob = sess.run(vgg16_b['prob'])[0]
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])

writer.add_graph(sess.graph)

with tf.name_scope('VGG16_c'):
    vgg16_c, new_vgg_scope = generate_VGG16(VGG16_weights_file,
                                scope="new_factory",
                                remove_top=False,
                                input_shape=(1, IMG_W, IMG_H, CHANNELS),
                                input_tensor=None,
                                apply_preprocess=True)
with tf.name_scope('VGG16_d'):    
    vgg16_d, _ = generate_VGG16(VGG16_weights_file,
                                scope=new_vgg_scope,
                                remove_top=False,
                                input_shape=(1, IMG_W, IMG_H, CHANNELS),
                                input_tensor=None,
                                apply_preprocess=True)

# we created two new VGG16 networks, but into a new VGG factory.
writer.add_graph(sess.graph)

writer.close()

