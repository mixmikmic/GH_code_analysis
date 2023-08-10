import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from scipy.io import loadmat
import tensorflow as tf

import math

# Read datasets
VMU = loadmat('Makeup_VMU.mat')
YMU = loadmat('Makeup_YMU.mat')
MIW = loadmat('Makeup_MIW.mat')

# reshape all input
def reshape_input(matrix):
    num_img = matrix.shape[1]
    VMU_test = np.reshape(matrix[:,0],(150,130,3),order="F")
    new_matrix = np.zeros((num_img,150,130,3),dtype=np.uint8) # it is important to set it uint8 for imshow to work
    for i in range(num_img):
        new_matrix[i] = np.reshape(matrix[:,i],(150,130,3),order="F")
    return new_matrix
    
X = np.concatenate((reshape_input(VMU['VMU_matrix']), reshape_input(YMU['YMU_matrix']), 
                    reshape_input(MIW['MIW_matrix'])), axis=0)

print "Total number of images:", X.shape[0], "of size:", X.shape[1:]
# show first and last image
f, axarr = plt.subplots(1, 3, figsize=(15,40))

axarr[0].imshow(X[0])
axarr[1].imshow(X[500])
axarr[2].imshow(X[-1])
# Hide x and y ticks
axarr[0].set_xticks([])
axarr[0].set_yticks([])
axarr[1].set_xticks([])
axarr[1].set_yticks([])
axarr[2].set_xticks([])
axarr[2].set_yticks([])

# transform pixel values to make them between -0.5 and 0.5
pixel_depth = 255.0  # Number of levels per pixel.
X = (X.astype(float) - pixel_depth / 2) / pixel_depth

# see if transform them back gives correct result
f, axarr = plt.subplots(1, 2, figsize=(10,40))

axarr[0].imshow(X[0])
axarr[0].set_title('transformed for tensorflow')
axarr[1].imshow((X[0]*pixel_depth + pixel_depth / 2).astype(np.uint8))
axarr[1].set_title('transformed back')
# Hide x and y ticks
axarr[0].set_xticks([])
axarr[0].set_yticks([])
axarr[1].set_xticks([])
axarr[1].set_yticks([])

# Identify data for which makeup/no-makeup pairs are available
# all files with makeup applications
indices = []
for i,iname in enumerate(VMU['VMU_filenames'][0]):
    if iname[0][-7:] == '_mu.jpg':
        indices += [i]
        
print 'number of images with makeup:',len(indices)

# Create labeled dataset of lipstick (X) and no lipstick (Y)
X_train = X[np.array(indices[:-3])-1]
Y_train = X[np.array(indices[:-3])-3]

X_test = X[np.array(indices[-3:])-1]
Y_test = X[np.array(indices[-3:])-3]

print X_train.shape, X_test.shape

# Image 48 was mislabeled
if X_train.shape[0] > 48:
    Y_train[48] = X[indices[48]+1]
else:
    Y_test[48-X_train.shape[0]] = X[indices[48]+1]

numofimg = 10
f, axarr = plt.subplots(numofimg, 2, figsize=(10,30))
for i in range(numofimg):
    axarr[i,0].imshow((X_train[i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    axarr[i,1].imshow((Y_train[i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    axarr[0,0].set_title('inputs')
    axarr[0,1].set_title('labels')
    # Hide x and y ticks
    axarr[i,0].set_xticks([])
    axarr[i,0].set_yticks([])
    axarr[i,1].set_xticks([])
    axarr[i,1].set_yticks([])

# Adjust color
for i in range(X_train.shape[0]):
    correction = np.mean(X_train[i,75] - Y_train[i,75],axis=0)
    for j in range(3):
        Y_train[i,:,:,j] += correction[j]
    
for i in range(X_test.shape[0]):
    correction = np.mean(X_test[i,75] - Y_test[i,75],axis=0)
    for j in range(3):
        Y_test[i,:,:,j] += correction[j]
    
numofimg = 10
f, axarr = plt.subplots(numofimg, 2, figsize=(10,30))
for i in range(numofimg):
    axarr[i,0].imshow((X_train[i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    axarr[i,1].imshow((Y_train[i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    axarr[0,0].set_title('inputs')
    axarr[0,1].set_title('labels')
    # Hide x and y ticks
    axarr[i,0].set_xticks([])
    axarr[i,0].set_yticks([])
    axarr[i,1].set_xticks([])
    axarr[i,1].set_yticks([])

# From tensorflow libs.activations
def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# Code from tensorflow example of convolutional autoencoder with slight modifications
def autoencoder(input_shape=[None, X.shape[1], X.shape[2], X.shape[3]],
                n_filters=[1, 10, 10, 10],
                filter_sizes=[3, 3, 3, 3]):
    """Build a deep denoising autoencoder w/ tied weights.
    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    Raises
    ------
    ValueError
        Description
    """
    # %%
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')
    x_label = tf.placeholder(
        tf.float32, input_shape, name='x_label')

    current_input = x

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%f
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x_label))

    # %%
    return {'x': x, 'z': z, 'y': y, 'x_label': x_label, 'cost': cost}

# Fit autoencoder data

ae = autoencoder(n_filters=[1, 120, 120, 120],filter_sizes=[3, 3, 3, 3])

# %%
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])


# %%
# We create a session to use the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# %%
# Fit all training data for autoencoder
batch_size = 2
n_epochs = 200
for epoch_i in range(n_epochs):
    for batch_i in range(X.shape[0] // batch_size):
        batch_xs = X[batch_i*batch_size:batch_i*batch_size + batch_size]
        sess.run(optimizer, feed_dict={ae['x']: batch_xs, ae['x_label']: batch_xs})
    print epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: batch_xs, ae['x_label']: batch_xs})
    

# %%
# Plot example reconstructions
n_examples = 10
test_xs = X_train[:n_examples]
recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs})
print recon.shape


f, axarr = plt.subplots(n_examples, 2, figsize=(10,30))
axarr[0,0].set_title('input')
axarr[0,1].set_title('guess')
for example_i in range(n_examples):
    axarr[example_i,0].imshow((X_train[example_i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    axarr[example_i,1].imshow((recon[example_i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    
    # Hide x and y ticks
    axarr[example_i,0].set_xticks([])
    axarr[example_i,0].set_yticks([])
    axarr[example_i,1].set_xticks([])
    axarr[example_i,1].set_yticks([])

# Fit lipstick data in the same session
n_epochs = 50
for epoch_i in range(n_epochs):
    for batch_i in range(X_train.shape[0] // batch_size):
        batch_xs = X_train[batch_i*batch_size:batch_i*batch_size + batch_size]
        batch_xlabel = Y_train[batch_i*batch_size:batch_i*batch_size + batch_size]
        sess.run(optimizer, feed_dict={ae['x']: batch_xs, ae['x_label']: batch_xlabel})
    print epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: batch_xs, ae['x_label']: batch_xlabel})
    
# %%
# Plot example reconstructions
n_examples = 10
test_xs = X_train[:n_examples]
recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs})
print recon.shape


f, axarr = plt.subplots(n_examples, 3, figsize=(10,40))
axarr[0,0].set_title('input')
axarr[0,1].set_title('guess')
axarr[0,2].set_title('target')
for example_i in range(n_examples):
    axarr[example_i,0].imshow((X_train[example_i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    axarr[example_i,1].imshow((recon[example_i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    axarr[example_i,2].imshow((Y_train[example_i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    
    # Hide x and y ticks
    axarr[example_i,0].set_xticks([])
    axarr[example_i,0].set_yticks([])
    axarr[example_i,1].set_xticks([])
    axarr[example_i,1].set_yticks([])
    axarr[example_i,2].set_xticks([])
    axarr[example_i,2].set_yticks([])

# print test set
recon = sess.run(ae['y'], feed_dict={ae['x']: X_test})

n_examples = X_test.shape[0]

f, axarr = plt.subplots(n_examples, 3, figsize=(10,10))
axarr[0,0].set_title('input')
axarr[0,1].set_title('guess')
axarr[0,2].set_title('target')
for example_i in range(n_examples):
    axarr[example_i,0].imshow((X_test[example_i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    axarr[example_i,1].imshow((recon[example_i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    axarr[example_i,2].imshow((Y_test[example_i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    
    # Hide x and y ticks
    axarr[example_i,0].set_xticks([])
    axarr[example_i,0].set_yticks([])
    axarr[example_i,1].set_xticks([])
    axarr[example_i,1].set_yticks([])
    axarr[example_i,2].set_xticks([])
    axarr[example_i,2].set_yticks([])

# Try on unseen dataset (used for autoencoding)
recon = sess.run(ae['y'], feed_dict={ae['x']: X[[-17,-15]]})

f, axarr = plt.subplots(2, 2, figsize=(10,10))

for i,example_i in enumerate([-17,-15]):
    axarr[i,0].imshow((X[example_i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    axarr[i,1].imshow((recon[i]*pixel_depth + pixel_depth / 2).astype(np.uint8))
    
    # Hide x and y ticks
    axarr[i,0].set_xticks([])
    axarr[i,0].set_yticks([])
    axarr[i,1].set_xticks([])
    axarr[i,1].set_yticks([])

saver = tf.train.Saver()
# Save the variables to disk.
save_path = saver.save(sess, "./model0428.ckpt")
print "Model saved in file: %s" % save_path



