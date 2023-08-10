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

STYLE_IMAGE = 'images/udnie.jpg'

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

style_image = utils.load_image(STYLE_IMAGE,expand_dim=True)
print(style_image.shape)
plt.imshow(style_image[0])

mean_style_image = np.mean(style_image)
print(mean_style_image)

def style_loss(sess, model, layer):
    """
    Style loss function as defined in the paper.
    """
    def _gram_matrix(F, N, M):
        """
        The gram matrix G.
        """
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _loss(a, x):
        """
        The style loss calculation.
        """
        # N is the number of filters (at layer l).
        N = a.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = a.shape[1] * a.shape[2]
        # A is the style representation of the original image (at layer l) : numpy array
        A = _gram_matrix(a, N, M)
        # G is the style representation of the generated image (at layer l) : tensor
        G = _gram_matrix(x, N, M)
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    loss = _loss(sess.run(model[layer]), model[layer])
    
    return loss

def generate_noise_image():
    """
    Returns a noise image intermixed with the content image at a certain ratio.
    """
    noise_image = np.random.uniform(-20, 20, (1, IMG_H, IMG_W, CHANNELS)).astype('float32')
    return noise_image

input_image = generate_noise_image()
print(input_image.shape)
plt.imshow(input_image[0])

# Construct style_loss using content_image.
sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(style_image))

with tf.name_scope("style_loss"):
    loss = style_loss(sess, model, 'conv1_2')
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(loss)

input_image = generate_noise_image()
sess.run(tf.global_variables_initializer())
_ = sess.run(model['input'].assign(input_image))

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

get_ipython().run_cell_magic('time', '', "\nITERATIONS = 1000\n\nfor it in range(ITERATIONS):\n    _, summary = sess.run([train_step, merged])\n    writer.add_summary(summary, it)\n    if it%100 == 0:\n        _image = sess.run(model['input'])\n        print('Iteration %d' % (it))\n        print('cost: ', sess.run(loss))\n        filename = 'output/style_conv1_2_iter{}.png'.format(it)\n        utils.save_image(filename, _image)\n        \n_image = sess.run(model['input'])\nprint('Iteration %d' % (it))\nprint('cost: ', sess.run(loss))\nfilename = 'output/style_conv1_2_iter{}.png'.format(it)\nutils.save_image(filename, _image)")

_image = _image[0]
_image = _image + 0.7*(mean_style_image - np.mean(_image))
_image = np.clip(_image, 0, 255).astype('uint8')
plt.imshow(_image)

writer.close()

# Construct style_loss using content_image.
sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(style_image))

with tf.name_scope("style_loss"):
    loss = style_loss(sess, model, 'conv2_2')
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(loss)
    
input_image = generate_noise_image()
sess.run(tf.global_variables_initializer())
_ = sess.run(model['input'].assign(input_image))

get_ipython().run_cell_magic('time', '', "\nITERATIONS = 1000\n\nfor it in range(ITERATIONS):\n    sess.run(train_step)\n    if it%100 == 0:\n        _image = sess.run(model['input'])\n        print('Iteration %d' % (it))\n        print('cost: ', sess.run(loss))\n        filename = 'output/style_conv2_2_iter{}.png'.format(it)\n        utils.save_image(filename, _image)\n        \n_image = sess.run(model['input'])\nprint('Iteration %d' % (it))\nprint('cost: ', sess.run(loss))\nfilename = 'output/style_conv2_2_iter{}.png'.format(it)\nutils.save_image(filename, _image)")

_image = _image[0]
_image = _image + 0.7*(mean_style_image - np.mean(_image))
_image = np.clip(_image, 0, 255).astype('uint8')
plt.imshow(_image)

# Construct style_loss using content_image.
sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(style_image))

with tf.name_scope("style_loss"):
    loss = style_loss(sess, model, 'conv3_3')
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(loss)
    
input_image = generate_noise_image()
sess.run(tf.global_variables_initializer())
_ = sess.run(model['input'].assign(input_image))

get_ipython().run_cell_magic('time', '', "\nITERATIONS = 1000\n\nfor it in range(ITERATIONS):\n    sess.run(train_step)\n    if it%100 == 0:\n        _image = sess.run(model['input'])\n        print('Iteration %d' % (it))\n        print('cost: ', sess.run(loss))\n        filename = 'output/style_conv3_3_iter{}.png'.format(it)\n        utils.save_image(filename, _image)\n        \n_image = sess.run(model['input'])\nprint('Iteration %d' % (it))\nprint('cost: ', sess.run(loss))\nfilename = 'output/style_conv3_3_iter{}.png'.format(it)\nutils.save_image(filename, _image)")

_image = _image[0]
_image = _image + 0.7*(mean_style_image - np.mean(_image))
_image = np.clip(_image, 0, 255).astype('uint8')
plt.imshow(_image)

# Construct style_loss using content_image.
sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(style_image))

with tf.name_scope("style_loss"):
    loss = style_loss(sess, model, 'conv4_3')
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(loss)
    
input_image = generate_noise_image()
sess.run(tf.global_variables_initializer())
_ = sess.run(model['input'].assign(input_image))

get_ipython().run_cell_magic('time', '', "\nITERATIONS = 1000\n\nfor it in range(ITERATIONS):\n    sess.run(train_step)\n    if it%100 == 0:\n        _image = sess.run(model['input'])\n        print('Iteration %d' % (it))\n        print('cost: ', sess.run(loss))\n        filename = 'output/style_conv4_3_iter{}.png'.format(it)\n        utils.save_image(filename, _image)\n        \n_image = sess.run(model['input'])\nprint('Iteration %d' % (it))\nprint('cost: ', sess.run(loss))\nfilename = 'output/style_conv4_3_iter{}.png'.format(it)\nutils.save_image(filename, _image)")

_image = _image[0]
_image = _image + 0.7*(mean_style_image - np.mean(_image))
_image = np.clip(_image, 0, 255).astype('uint8')
plt.imshow(_image)

# Construct style_loss using content_image.
sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(style_image))

with tf.name_scope("style_loss"):
    loss = style_loss(sess, model, 'conv5_3')
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(loss)
    
input_image = generate_noise_image()
sess.run(tf.global_variables_initializer())
_ = sess.run(model['input'].assign(input_image))

get_ipython().run_cell_magic('time', '', "\nITERATIONS = 1000\n\nfor it in range(ITERATIONS):\n    sess.run(train_step)\n    if it%100 == 0:\n        _image = sess.run(model['input'])\n        print('Iteration %d' % (it))\n        print('cost: ', sess.run(loss))\n        filename = 'output/style_conv5_3_iter{}.png'.format(it)\n        utils.save_image(filename, _image)\n        \n_image = sess.run(model['input'])\nprint('Iteration %d' % (it))\nprint('cost: ', sess.run(loss))\nfilename = 'output/style_conv5_3_iter{}.png'.format(it)\nutils.save_image(filename, _image)")

_image = _image[0]
_image = _image + 0.7*(mean_style_image - np.mean(_image))
_image = np.clip(_image, 0, 255).astype('uint8')
plt.imshow(_image)





