# Further information
# Google Blog Entry https://research.googleblog.com/2016/08/improving-inception-and-image.html
# Models to Train   https://github.com/tensorflow/models/blob/master/slim/README.md#Tuning
# Transfer Learning http://stackoverflow.com/questions/40350539/tfslim-problems-loading-saved-checkpoint-for-vgg16

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
from imagenet_classes import class_names

tf.reset_default_graph()
images = tf.placeholder(tf.float32, [None, None, None, 3])
imgs_scaled = tf.image.resize_images(images, (224,224))

# Note we call the network studid, so that we don't get a hold for the tensors we need.
slim.nets.vgg.vgg_16(imgs_scaled, is_training=False)
variables_to_restore = slim.get_variables_to_restore()
print('Number of variables to restore {}'.format(len(variables_to_restore)))
init_assign_op, init_feed_dict = slim.assign_from_checkpoint('/Users/oli/Dropbox/server_sync/tf_slim_models/vgg_16.ckpt', variables_to_restore)
sess = tf.Session()
sess.run(init_assign_op, init_feed_dict)

img1 = imread('poodle.jpg')
print(img1.shape)
print("Some pixels {}".format(img1[199,199:205,0]))
plt.imshow(img1)
plt.show()

tf.train.SummaryWriter('/tmp/dumm/vgg16', tf.get_default_graph()).close()
#tensorboard --logdir /tmp/dumm/

ops = tf.get_default_graph().get_operations()
for i in ops[0:9]:
    print i.name

g = tf.get_default_graph()
feed = g.get_tensor_by_name('Placeholder:0')
fetch = g.get_tensor_by_name('vgg_16/fc8/BiasAdd:0')

feed = tf.Graph.get_tensor_by_name(tf.get_default_graph(), 'Placeholder:0')
fetch = tf.Graph.get_tensor_by_name(tf.get_default_graph(), 'vgg_16/fc8/BiasAdd:0')

feed.get_shape(), fetch.get_shape()

# Make a tensor of order 4
feed_vals = [img1]
np.shape(feed_vals)

res = sess.run(fetch, feed_dict={feed:feed_vals})

res.shape

d = res[0,0,0,]
prob = np.exp(d)/np.sum(np.exp(d))
preds = (np.argsort(prob)[::-1])[0:10]
for p in preds:
    print p, class_names[p], prob[p]

# First method use the variable tensor by name
g = tf.get_default_graph()
var = g.get_tensor_by_name("vgg_16/conv1/conv1_1/weights:0")
conv1_1_alt = sess.run(var)

# Second method use the variable scope mechanism and sharing
with tf.variable_scope("vgg_16/conv1/conv1_1", reuse=True):
    var = tf.get_variable("weights")
conv1_1 = sess.run(var)

# Third method (python magic). List all variables and pick the right one
var = [v for v in tf.all_variables() if v.name == 'vgg_16/conv1/conv1_1/weights:0'][0]
conv1_1_alt_3 = sess.run(var)

np.sum(conv1_1_alt != conv1_1), np.sum(conv1_1_alt != conv1_1_alt_3)

plt.imshow(conv1_1[0,:,:,0], interpolation='none', cmap=plt.get_cmap('gray'))

sess.close()

