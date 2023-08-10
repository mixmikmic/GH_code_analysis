import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

import utils
from vgg.imagenet_classes import class_names
from vgg.VGG import generate_VGG16

print("TensorFlow version : {}".format(tf.__version__))
print("Devices : {}".format(utils.get_tensorflow_devices()))

IMG_W = 224
IMG_H = 224
CHANNELS = 3

MODEL_WEIGHTS = 'vgg/vgg16.npy'

LOG_DIR = 'logs/'

CONTENT_IMAGE = 'images/golden_retriever.jpg'

if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
    
tf.gfile.MakeDirs(LOG_DIR)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

model, vgg_scope = generate_VGG16(weights_file=MODEL_WEIGHTS,
                                  scope="VGG16",
                                  apply_preprocess=True,
                                  remove_top=True,
                                  input_shape=(1, IMG_W, IMG_H, CHANNELS))

content_image = utils.load_image(CONTENT_IMAGE,expand_dim=True)
print(content_image.shape)
plt.imshow(content_image[0])

def content_loss(sess, model, layer):
    
    def _loss(p, x):
        # N is the number of filters (at layer l).
        N = p.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = p.shape[1] * p.shape[2]
        return (1 / (4 * M * N)) * tf.reduce_sum(tf.pow(x - p, 2))
    
    return _loss(sess.run(model[layer]), model[layer])

def generate_noise_image():
    """
    Returns a noise image intermixed with the content image at a certain ratio.
    """
    noise_image = np.random.uniform(-20, 20, (1, IMG_H, IMG_W, CHANNELS)).astype('float32')
    return noise_image

input_image = generate_noise_image()
print(input_image.shape)
plt.imshow(input_image[0])

# Construct content_loss using content_image.
sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(content_image))

with tf.name_scope("content_loss"):
    loss = content_loss(sess, model, 'conv1_2')
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(loss)

input_image = generate_noise_image()
sess.run(tf.global_variables_initializer())
_ = sess.run(model['input'].assign(input_image))
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

get_ipython().run_cell_magic('time', '', "\nITERATIONS = 4000\n\nfor it in range(ITERATIONS):\n    _, summary = sess.run([train_step, merged])\n    writer.add_summary(summary, it)\n    if it%500 == 0:\n        _image = sess.run(model['input'])\n        print('Iteration %d' % (it))\n        print('cost: ', sess.run(loss))\n        filename = 'output/conv1_2_iter{}.png'.format(it)\n        utils.save_image(filename, _image)\n        \n        \n_image = sess.run(model['input'])\nprint('Iteration %d' % (it))\nprint('cost: ', sess.run(loss))\nfilename = 'output/conv1_2_iter{}.png'.format(it)\nutils.save_image(filename, _image)")

_image = _image[0]
_image = np.clip(_image, 0, 255).astype('uint8')
plt.imshow(_image)

# stop tensorboard monitoring
writer.close()

# Construct content_loss using content_image.
sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(content_image))

with tf.name_scope("content_loss"):
    loss = content_loss(sess, model, 'conv2_2')
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(loss)

input_image = generate_noise_image()
sess.run(tf.global_variables_initializer())
_ = sess.run(model['input'].assign(input_image))

get_ipython().run_cell_magic('time', '', "\nITERATIONS = 4000\n\nfor it in range(ITERATIONS):\n    sess.run(train_step)\n    if it%500 == 0:\n        _image = sess.run(model['input'])\n        print('Iteration %d' % (it))\n        print('cost: ', sess.run(loss))\n        filename = 'output/conv2_2_iter{}.png'.format(it)\n        utils.save_image(filename, _image)\n        \n_image = sess.run(model['input'])\nprint('Iteration %d' % (it))\nprint('cost: ', sess.run(loss))\nfilename = 'output/conv2_2_iter{}.png'.format(it)\nutils.save_image(filename, _image)")

_image = _image[0]
_image = np.clip(_image, 0, 255).astype('uint8')
plt.imshow(_image)

# Construct content_loss using content_image.
sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(content_image))

with tf.name_scope("content_loss"):
    loss = content_loss(sess, model, 'conv3_3')
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(loss)

input_image = generate_noise_image()
sess.run(tf.global_variables_initializer())
_ = sess.run(model['input'].assign(input_image))

get_ipython().run_cell_magic('time', '', "\nITERATIONS = 10000\n\nfor it in range(ITERATIONS):\n    sess.run(train_step)\n    if it%1000 == 0:\n        _image = sess.run(model['input'])\n        print('Iteration %d' % (it))\n        print('cost: ', sess.run(loss))\n        filename = 'output/conv3_3_iter{}.png'.format(it)\n        utils.save_image(filename, _image)\n        \n_image = sess.run(model['input'])\nprint('Iteration %d' % (it))\nprint('cost: ', sess.run(loss))\nfilename = 'output/conv3_3_iter{}.png'.format(it)\nutils.save_image(filename, _image)")

_image = _image[0]
_image = np.clip(_image, 0, 255).astype('uint8')
plt.imshow(_image)

# Construct content_loss using content_image.
sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(content_image))

with tf.name_scope("content_loss"):
    loss = content_loss(sess, model, 'conv4_3')
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(loss)

input_image = generate_noise_image()
sess.run(tf.global_variables_initializer())
_ = sess.run(model['input'].assign(input_image))

get_ipython().run_cell_magic('time', '', "\nITERATIONS = 10000\n\nfor it in range(ITERATIONS):\n    sess.run(train_step)\n    if it%1000 == 0:\n        _image = sess.run(model['input'])\n        print('Iteration %d' % (it))\n        print('cost: ', sess.run(loss))\n        filename = 'output/conv4_3_iter{}.png'.format(it)\n        utils.save_image(filename, _image)\n        \n_image = sess.run(model['input'])\nprint('Iteration %d' % (it))\nprint('cost: ', sess.run(loss))\nfilename = 'output/conv4_3_iter{}.png'.format(it)\nutils.save_image(filename, _image)")

_image = _image[0]
_image = np.clip(_image, 0, 255).astype('uint8')
plt.imshow(_image)

# Construct content_loss using content_image.
sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(content_image))

with tf.name_scope("content_loss"):
    loss = content_loss(sess, model, 'conv5_3')
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(loss)

input_image = generate_noise_image()
sess.run(tf.global_variables_initializer())
_ = sess.run(model['input'].assign(input_image))

get_ipython().run_cell_magic('time', '', "\nITERATIONS = 10000\n\nfor it in range(ITERATIONS):\n    sess.run(train_step)\n    if it%1000 == 0:\n        _image = sess.run(model['input'])\n        print('Iteration %d' % (it))\n        print('cost: ', sess.run(loss))\n        filename = 'output/conv5_3_iter{}.png'.format(it)\n        utils.save_image(filename, _image)\n        \n        \n_image = sess.run(model['input'])\nprint('Iteration %d' % (it))\nprint('cost: ', sess.run(loss))\nfilename = 'output/conv5_3_iter{}.png'.format(it)\nutils.save_image(filename, _image)")

_image = _image[0]
_image = np.clip(_image, 0, 255).astype('uint8')
plt.imshow(_image)





