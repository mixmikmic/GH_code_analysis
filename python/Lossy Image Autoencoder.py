get_ipython().magic('pylab inline')

import multiprocessing
import tensorflow as tf

from numba import jit
from keras.datasets import cifar10
from skimage.color import rgb2gray

from sklearn.decomposition import PCA

# Set random seed (for reproducibility)
np.random.seed(1000)

width = 32
height = 32
batch_size = 10
nb_epochs = 15
code_length = 128

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

graph = tf.Graph()

with graph.as_default():
    # Global step
    global_step = tf.Variable(0, trainable=False)
    
    # Input batch
    input_images = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))

    # Convolutional layer 1
    conv1 = tf.layers.conv2d(inputs=input_images,
                             filters=32,
                             kernel_size=(3, 3),
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=tf.nn.tanh)

    # Convolutional output (flattened)
    conv_output = tf.contrib.layers.flatten(conv1)

    # Code layer
    code_layer = tf.layers.dense(inputs=conv_output,
                                 units=code_length,
                                 activation=tf.nn.tanh)
    
    # Code output layer
    code_output = tf.layers.dense(inputs=code_layer,
                                  units=(height - 2) * (width - 2) * 3,
                                  activation=tf.nn.tanh)

    # Deconvolution input
    deconv_input = tf.reshape(code_output, (batch_size, height - 2, width - 2, 3))

    # Deconvolution layer 1
    deconv1 = tf.layers.conv2d_transpose(inputs=deconv_input,
                                         filters=3,
                                         kernel_size=(3, 3),
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         activation=tf.sigmoid)
    
    # Output batch
    output_images = tf.cast(tf.reshape(deconv1, 
                                       (batch_size, height, width, 3)) * 255.0, tf.uint8)

    # Reconstruction L2 loss
    loss = tf.nn.l2_loss(input_images - deconv1)

    # Training operations
    learning_rate = tf.train.exponential_decay(learning_rate=0.0005, 
                                               global_step=global_step, 
                                               decay_steps=int(X_train.shape[0] / (2 * batch_size)), 
                                               decay_rate=0.95, 
                                               staircase=True)
    
    trainer = tf.train.RMSPropOptimizer(learning_rate)
    training_step = trainer.minimize(loss)

use_gpu = False

config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(), 
                        inter_op_parallelism_threads=multiprocessing.cpu_count(), 
                        allow_soft_placement=True, 
                        device_count = {'CPU' : 1, 
                                        'GPU' : 1 if use_gpu else 0})

session = tf.InteractiveSession(graph=graph, config=config)

tf.global_variables_initializer().run()

@jit
def create_batch(t, gray=False):
    X = np.zeros((batch_size, height, width, 3 if not gray else 1), dtype=np.float32)
        
    for k, image in enumerate(X_train[t:t+batch_size]):
        if gray:
            X[k, :, :, :] = rgb2gray(image)
        else:
            X[k, :, :, :] = image / 255.0
        
    return X

for e in range(nb_epochs):
    total_loss = 0.0
    
    for t in range(0, X_train.shape[0], batch_size):
        feed_dict = {
            input_images: create_batch(t)
        }

        _, v_loss = session.run([training_step, loss], feed_dict=feed_dict)
        total_loss += v_loss
        
    print('Epoch {} - Total loss: {}'.format(e+1, total_loss))

feed_dict = {
    input_images: create_batch(0)
}


oimages = session.run([output_images], feed_dict=feed_dict)

fig, ax = plt.subplots(2, batch_size, figsize=(18, 3))

for y in range(batch_size):
    ax[0, y].get_xaxis().set_visible(False)
    ax[0, y].get_yaxis().set_visible(False)
    ax[1, y].get_xaxis().set_visible(False)
    ax[1, y].get_yaxis().set_visible(False)

    ax[0, y].imshow(X_train[y])
    ax[1, y].imshow(oimages[0][y])



