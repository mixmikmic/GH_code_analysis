__author__ = 'Long Jin, Weijian Xu, and Kwonjoon Lee'

import os
import sys
import time
import copy
import scipy
import sklearn
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from utils import *
from numpy import inf

# Display the versions for libraries. In my environment, they are
#     Python version: 2.7.13 |Anaconda custom (64-bit)| (default, Sep 30 2017, 18:12:43)
#     [GCC 7.2.0]
#     SciPy version: 0.19.1
#     NumPy version: 1.14.2
#     TensorFlow version: 1.7.0
#     Scikit-learn version: 0.19.0
print('Python version: {}'.format(sys.version))
print('SciPy version: {}'.format(scipy.__version__))
print('NumPy version: {}'.format(np.__version__))
print('TensorFlow version: {}'.format(tf.__version__))
print('Scikit-learn version: {}'.format(sklearn.__version__))

import functools
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.small_imagenet
import tflib.ops.layernorm
import tflib.plot

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

# Batch size. It should be a squared number.
batch_size = 100
# Number of cascades in WINN.
cascades = 4
# Number of iterations per cascade
# In Algorithm 1 of the paper, we wrote we iterate while W_t has not converged.
# In practice, we find that 100 iterations is sufficient for convergence.
iterations_per_cascade = 100
# hyperparameter k in the Algorithm 1
k = 3

# Root directory of data directory. Customize it when using another directory.
# e.g. "./"
data_dir_root = "/mnt/cube/kwl042/church_release_candidate_3"
# Path of data directory.
data_dir_path = os.path.join(data_dir_root, "data")

# Create a series of directories to contain the dataset.
mkdir_if_not_exists(data_dir_root)
mkdir_if_not_exists(data_dir_path)

# Path of training images directory.
training_images_dir_path = "/mnt/cube/kwl042/church/"

def swish(z):
    return z * tf.sigmoid(z)

def Normalize(name, axes, inputs):
    return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def GoodDiscriminator(inputs, dim=64, nonlinearity = swish, bn = True, reuse = False):
    output = tf.reshape(tf.transpose(inputs, [0, 3, 1, 2]), [-1, 3, 64, 64])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)
    
    with tf.variable_scope("layers", reuse = reuse):
        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, 32, 3, output, stride=1, he_init=False)
        output = nonlinearity(output)
        
        output = lib.ops.conv2d.Conv2D('Discriminator.2', 32, 64, 3, output, stride=1, he_init=False)
        if bn:
            output = Normalize('Discriminator.BN2', [0,2,3], output)
        output = nonlinearity(output)
        
        output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
        ### output: 64 channels x 32 x 32
        
        output = lib.ops.conv2d.Conv2D('Discriminator.3', 64, 64, 3, output, stride=1, he_init=False)
        if bn:
            output = Normalize('Discriminator.BN3', [0,2,3], output)
        output = nonlinearity(output)
        
        output = lib.ops.conv2d.Conv2D('Discriminator.4', 64, 128, 3, output, stride=1, he_init=False)
        if bn:
            output = Normalize('Discriminator.BN4', [0,2,3], output)
        output = nonlinearity(output)
        
        output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
        ### output: 128 channels x 16 x 16

        output = lib.ops.conv2d.Conv2D('Discriminator.5', 128, 128, 3, output, stride=1, he_init=False)
        if bn:
            output = Normalize('Discriminator.BN5', [0,2,3], output)
        output = nonlinearity(output)
        
        output = lib.ops.conv2d.Conv2D('Discriminator.6', 128, 256, 3, output, stride=1, he_init=False)
        if bn:
            output = Normalize('Discriminator.BN6', [0,2,3], output)
        output = nonlinearity(output)
        
        output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
        ### output: 256 channels x 8 x 8
        
        output = lib.ops.conv2d.Conv2D('Discriminator.7', 256, 256, 3, output, stride=1, he_init=False)
        if bn:
            output = Normalize('Discriminator.BN7', [0,2,3], output)
        output = nonlinearity(output)
        
        output = lib.ops.conv2d.Conv2D('Discriminator.8', 256, 512, 3, output, stride=1, he_init=False)
        if bn:
            output = Normalize('Discriminator.BN8', [0,2,3], output)
        output = nonlinearity(output)
        
        output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
        ### output: 512 channels x 4 x 4
        
        output = tf.reshape(output, [-1, 4*4*512])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*512, 1, output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1])

def NoiseProvider(n_samples, noise=None, dim=64):
    lib.ops.conv2d.set_weights_stdev(0.1)
    lib.ops.deconv2d.set_weights_stdev(0.1)
    lib.ops.linear.set_weights_stdev(0.1)
    with tf.variable_scope("layers_np", reuse = False):
        output = noise
        output = lib.ops.conv2d.Conv2D('NoiseProvider.2', 8*dim, 4*dim, 5, output, stride=1)
        output = tf.transpose(output, [0, 2, 3, 1])
        output = tf.image.resize_images(output, [8, 8], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        output = tf.transpose(output, [0, 3, 1, 2])
        output = Normalize('NoiseProvider.BN2', [0,2,3], output)

        output = lib.ops.conv2d.Conv2D('NoiseProvider.3', 4*dim, 2*dim, 5, output, stride=1)
        output = tf.transpose(output, [0, 2, 3, 1])
        output = tf.image.resize_images(output, [16, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        output = tf.transpose(output, [0, 3, 1, 2])
        output = Normalize('NoiseProvider.BN3', [0,2,3], output)

        output = lib.ops.conv2d.Conv2D('NoiseProvider.4', 2*dim, dim, 5, output, stride=1)
        output = tf.transpose(output, [0, 2, 3, 1])
        output = tf.image.resize_images(output, [32, 32], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        output = tf.transpose(output, [0, 3, 1, 2])
        output = Normalize('NoiseProvider.BN4', [0,2,3], output)

        output = lib.ops.conv2d.Conv2D('NoiseProvider.5', dim, 3, 5, output, stride=1)
        output = tf.transpose(output, [0, 2, 3, 1])
        output = tf.image.resize_images(output, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        output = tf.transpose(output, [0, 3, 1, 2])

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

    return output

def build_network(batch_shape, LAMBDA=10.0):
    '''
    Build a network for WINN.
        batch_shape: Shape of a mini-batch in classification-step and synthesis-step.
                     The format is [batch size, height, width, channels].
        LAMBDA: the weight for the gradient penalty term
    Return loss, trainable variables, labels and images in discriminator and 
    sampler, plus checkpoint saver. 
    '''

    # Fetch batch shape.
    [batch_size, height, width, channels] = batch_shape
    
    half_b_size = batch_size / 2
    
    # Placeholder for images and labels.
    D_pos_images = tf.placeholder(dtype = tf.float32, 
                              shape = [half_b_size, height, width, channels], 
                              name = 'D_pos_images')
    D_neg_images = tf.placeholder(dtype = tf.float32, 
                              shape = [half_b_size, height, width, channels], 
                              name = 'D_neg_images')
    
    # Variable, placeholder and assign operator for multiple sampled images.
    S_images = tf.Variable(
        # Use uniform distribution Unif(-1, 1) to initialize.
        # This initialization doesn't matter.
        # It will be substituted by S_images_op.
        np.random.uniform(low = -1.0,
                          high = 1.0, 
                          size = [batch_size, height, width, channels]
        ).astype('float32'), 
        name='S_images'
    )
    S_images_placeholder = tf.placeholder(dtype = S_images.dtype, 
                                          shape = S_images.get_shape())
    S_images_op = S_images.assign(S_images_placeholder)

    # Build a discriminator used in classification-step
    D_pos_logits = GoodDiscriminator(D_pos_images, reuse = False)
    D_neg_logits = GoodDiscriminator(D_neg_images, reuse = True)
    D_loss = tf.reduce_mean(D_neg_logits - D_pos_logits)
    D_pos_loss = tf.reduce_mean(D_pos_logits)
    epsilon = tf.random_uniform([half_b_size, 1, 1, 1], 0.0, 1.0)
    # Dirty hack to tile the tensor
    epsilon = epsilon + tf.zeros(D_pos_images.shape, dtype=epsilon.dtype)
    x_hat = epsilon * D_pos_images + (1 - epsilon) * D_neg_images
    d_hat = GoodDiscriminator(x_hat, reuse = True)
    
    ddx = tf.gradients(d_hat, x_hat)[0]
    ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=[1, 2, 3]))
    ddx = tf.reduce_mean(tf.square(ddx - 1.0) * LAMBDA)
    D_loss += ddx
    
    # We need to store these values as they will be used for determining early-stopping threshold in testing stage
    D_pos_loss_min = tf.Variable(0.0, name='D_pos_loss_min')
    D_pos_loss_max = tf.Variable(0.0, name='D_pos_loss_max')
    
    D_pos_loss_min_placeholder = tf.placeholder(dtype = D_pos_loss_min.dtype, 
                                          shape = D_pos_loss_min.get_shape())
    D_pos_loss_max_placeholder = tf.placeholder(dtype = D_pos_loss_max.dtype, 
                                          shape = D_pos_loss_max.get_shape())
    D_pos_loss_min_op = D_pos_loss_min.assign(D_pos_loss_min_placeholder)
    D_pos_loss_max_op = D_pos_loss_max.assign(D_pos_loss_max_placeholder)

    # Build a sampler used in synthesis-step
    S_logits = GoodDiscriminator(S_images, reuse = True)
    S_loss = tf.reduce_mean(S_logits)

    # Variable, placeholder and assign operator for multiple generated images.
    small_noise = tf.Variable(
        np.random.uniform(low = -1.0,
                          high = 1.0, 
                          size = [batch_size, 512, 4, 4]
        ).astype('float32'),
        name='small_noise'
    )
    small_noise_placeholder = tf.placeholder(dtype = small_noise.dtype, 
                                          shape = small_noise.get_shape())
    small_noise_op = small_noise.assign(small_noise_placeholder)
    
    big_noise = NoiseProvider(100, noise=small_noise, dim=64)
    # Variables to train.
    trainable_vars = tf.trainable_variables()
    D_vars = [var for var in trainable_vars if 'layers' in var.name]
    S_vars = [var for var in trainable_vars if 'S_images' in var.name]
    
    # Checkpoint saver.
    saver = tf.train.Saver(max_to_keep = 5000)
    
    return [D_loss, S_loss, D_vars, S_vars, 
            D_pos_images, D_neg_images, S_images, S_images_op, S_images_placeholder,
            saver, D_pos_loss, D_pos_loss_min, D_pos_loss_max,
            D_pos_loss_min_placeholder, D_pos_loss_max_placeholder,
            D_pos_loss_min_op, D_pos_loss_max_op, 
            small_noise, small_noise_op, small_noise_placeholder, big_noise]

# From https://github.com/Mazecreator/tensorflow-hints/tree/master/maximize
def maximize(optimizer, loss, **kwargs):
      return optimizer.minimize(-loss, **kwargs)

def get_optimizers(D_loss, S_loss, D_vars, S_vars):
    '''
    Get optimizers.
        D_loss: Discriminator loss.
        S_loss: Sampler loss.
        D_vars: Variables to train in discriminator.
        S_vars: Variable to train in sampler = image.
    Return optimizer of discriminator and sampler, plus discriminator 
    learning rate, discriminator global steps and the initializer for sampler.
    '''
    
    # Scope of discriminator optimizer.
    with tf.variable_scope('D_optimizer'):
        # Global count of step in discriminator and increment operator.
        # It should not be trainable and should be adjusted by training process.
        D_global_step = tf.Variable(initial_value = 0, trainable = False)
        D_global_step_op = D_global_step.assign_add(1)
        # Learning rate with exponential decay.
        D_learning_rate = tf.train.exponential_decay(learning_rate = 0.0001,
                                                     global_step = D_global_step,
                                                     decay_steps = 100,
                                                     decay_rate = 0.9,
                                                     staircase = True)        
        D_adam = tf.train.AdamOptimizer(learning_rate = D_learning_rate, beta1=0., beta2=0.9)
        D_optimizer = D_adam.minimize(loss = D_loss, var_list = D_vars)
        
    # Scope of sampler optimizer.
    with tf.variable_scope('S_optimizer'):
        S_global_step = tf.Variable(initial_value = 0, trainable = False, name = 'S_step')
        S_learning_rate = 0.01

        S_adam = tf.train.AdamOptimizer(learning_rate = S_learning_rate, beta1 = 0.9)
        S_optimizer = maximize(optimizer = S_adam, loss = S_loss, var_list = S_vars)
        
    # Variables of sampler optimizer and initializer operator of that.
    S_optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                         scope = 'WINN/S_optimizer')
    print ("S_optimizer_vars", S_optimizer_vars)
    S_initializer_op = tf.variables_initializer(S_optimizer_vars)

    # Variables of sampler optimizer and initializer operator of that.
    print(S_optimizer_vars)

    return [D_optimizer, S_optimizer, D_learning_rate, S_initializer_op, S_global_step]

def train(sess):
    """
    Train the WINN model.
        sess: Session.
    """

    # Set timer.
    start_time = time.time()
    
    half_batch_size = batch_size // 2
    sqrt_batch_size = int(np.sqrt(batch_size))

    # Log file path.
    log_file_path = os.path.join(data_dir_path, "log.txt")
    # Prepare for root directory of model.
    model_root = os.path.join(data_dir_path, "model")
    mkdir_if_not_exists(model_root)
    # Prepare for root directory of intermediate image.
    intermediate_image_root = os.path.join(data_dir_path, "intermediate")
    mkdir_if_not_exists(intermediate_image_root)
    # Prepare for root directory of negative images.
    neg_image_root = os.path.join(data_dir_path, "negative")
    mkdir_if_not_exists(neg_image_root)
        
    ######################################################################
    # Training stage 1: Load positive images.
    ######################################################################
    log(log_file_path,
        "Training stage 1: Load positive images...")

    # Path of all positive images and negative images. 
    # The following training_images_dir_path can be replaced. The image shape of
    # positive and negative images are the same.
    pos_all_images_path = get_images_path_in_directory(training_images_dir_path)
    image_shape = get_image_shape(pos_all_images_path[0])

    ######################################################################
    # Training stage 2: Build network and initialize.
    ######################################################################
    log(log_file_path,
        "Training stage 2: Build network and initialize...")
    height, width, channels = image_shape
    
    # Build network.
    [D_loss, S_loss, D_vars, S_vars, 
     D_pos_images, D_neg_images, S_images, S_images_op, S_images_placeholder,
     saver, D_pos_loss, D_pos_loss_min, D_pos_loss_max,
     D_pos_loss_min_placeholder, D_pos_loss_max_placeholder,
     D_pos_loss_min_op, D_pos_loss_max_op, 
     small_noise, small_noise_op, small_noise_placeholder, big_noise] = \
        build_network(batch_shape = [batch_size, height, width, channels])
        
    # Get optimizer.
    [D_optimizer, S_optimizer, D_learning_rate, S_initializer_op, S_global_step] =         get_optimizers(D_loss = D_loss, S_loss = S_loss, 
                       D_vars = D_vars, S_vars = S_vars)
        
    # Show a list of global variables.
    global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='')
    log(log_file_path, "Global variables:")
    for i, var in enumerate(global_variables):
        log(log_file_path, "{0} {1}".format(i, var.name))
        
    # Initialize all variables.
    all_initializer_op = tf.global_variables_initializer()
    sess.run(all_initializer_op)
    
    # Generate initial pseudo-negative images
    # In fact, the name of image has format 
    #     {cascade}_{next iteration}_{i}.png
    # where cascade means current cascade model, next iteration means
    # next iteration of sampler and discriminator training, and i means
    # the index of images.
    neg_image_path = os.path.join(neg_image_root, 'cascade_{0}_iteration_{1}_count_{2}.png')
    neg_init_images_count = 10000
    neg_init_images_path = [neg_image_path.format(0, 0, i)                             for i in range(neg_init_images_count)]
    
    S_iteration_count_of_batch = neg_init_images_count // batch_size
    
    for i in xrange(S_iteration_count_of_batch):             
        small_noise_batch = np.random.uniform(low=-1.0, high=1.0, size=(100, 512, 4, 4))
        sess.run(small_noise_op, {small_noise_placeholder: 
                       small_noise_batch})
        np_noise_images = np.transpose(sess.run(big_noise), axes=[0, 2, 3, 1])

        # Generate random images as negatives and save them.
        for j in range(100):#, neg_init_image_path in enumerate(neg_init_images_path):
            # Attention: Though it is called neg_image here, it has 4 dimensions,
            #            that is, [1, height, width, channels], which is not a
            #            pure single image, which is [height, width, channels].
            #            So we still use save_unnormalized_images here instead of 
            #            save_unnormalized_image.
            neg_image = np_noise_images[j].reshape(1, 64, 64, 3)
            neg_image = neg_image - neg_image.min()
            neg_image = neg_image / neg_image.max() * 255.0 
            save_unnormalized_images(images = neg_image, 
                                     size = (1, 1), path = neg_init_images_path[batch_size * i + j])

    neg_all_images_path = neg_init_images_path
    
    ######################################################
    log(log_file_path,
        "Positive images {0}, negative images {1}, image shape {2}".format(
        len(pos_all_images_path), len(neg_all_images_path), image_shape))
    
    ######################################################################
    # Training stage 3: Cascades training.
    ######################################################################
    log(log_file_path, "Training stage 3: Cascades training...")
        
    # One cascade means one new model.
    # p_{W^n}^- <- [cascade n] <- [cascade n-1] <- ... <- [cascade 0] <- p_r 
    # where p_r means Uniform or Gaussian reference distribution 
    # and p_{W^n}^- means distribution of pseudo-negatives after n cascades.
    # One cascade training consists multiple iterations (by default 100 iterations).
    
    # Prepare for the initial images to feed the sampler. In fact, it is 
    # because we always use negative images in last cascade as the "initial"
    # images to feed sampler in all iterations of current cascade.
    S_neg_last_cascade_images_path = copy.deepcopy(neg_all_images_path)
    
    for cascade in xrange(cascades):
        ######################################################################
        # Training stage 3.1: Iterations training.
        ######################################################################
        # One iteration means one time of discriminator training and one time
        # of sampling pseudo-negatives. One iteration training may contain multiple
        # batches for discriminator training and sampling pseudo-negatives.
        for iteration in xrange(iterations_per_cascade):
            ######################################################################
            # Training stage 3.1.1: Prepare images and labels for discriminator
            # training.
            ######################################################################
            # Count of positive images to train in current iteration.
            D_pos_iteration_images_count = min(iteration + 1, 5) * 1000                 // half_batch_size * half_batch_size
            
            if D_pos_iteration_images_count >= len(pos_all_images_path):
                # When the number of all positive images is more than current
                # iteration negative images, we allow duplicate images.
                D_pos_iteration_images_path = np.random.choice(
                    pos_all_images_path, 
                    size = D_pos_iteration_images_count, 
                    replace = True
                ).tolist()
            else:
                # When the number of all positive images is less or equal than 
                # current iteration negative images, we require unique images.
                D_pos_iteration_images_path = np.random.choice(
                    pos_all_images_path, 
                    size = D_pos_iteration_images_count, 
                    replace = False
                ).tolist()

            # Here we consider the "save all" mode in Long Jin's code. This mode
            # has different behaviors on discriminator and sampler.
            # 1) Discriminator.
            #     We draw positive images from training dataset and the same
            #     number of negative images from *all* pseudo-negative images in data/negative folder.
            #     Every iteration of sampler will add newly generated negative images
            #     into all data/negative foler.
            # 2) Sampler.
            #     We draw "initial" negative images in every iterations in current
            #     cascade from part of generated negative images in last cascade.
            #     More specificially, the part is the *last* iteration of last cascade.
            D_neg_iteration_images_count = D_pos_iteration_images_count
            D_neg_iteration_images_path = np.random.choice(
                neg_all_images_path,
                D_pos_iteration_images_count, 
                replace = True).tolist()
                            
            log(log_file_path,
                   ("Discriminator: Cascade {0}, iteration {1}, " + 
                   "all pos {2}, all neg {3}, " + 
                   "current iteration {4} (pos {5}, neg {6}), " + 
                   "learning rate {7}").format(
                       cascade, iteration, 
                       len(pos_all_images_path), len(neg_all_images_path), 
                       D_pos_iteration_images_count + D_neg_iteration_images_count, 
                       D_pos_iteration_images_count, D_neg_iteration_images_count, 
                       sess.run(D_learning_rate)
                   ))
            
            ######################################################################
            # Training stage 3.1.2: Train the discriminator.
            ######################################################################
            # Count of batch in discriminator training in current iteration. 
            D_iteration_count_of_batch = len(D_pos_iteration_images_path) // half_batch_size
            
            min_D_batch_pos_loss = inf
            max_D_batch_pos_loss = -inf
            
            for nc in range(k):
                for i in xrange(D_iteration_count_of_batch):
                    # Load images for this batch in discriminator.
                    D_pos_batch_images = [load_unnormalized_image(path) for path in
                        D_pos_iteration_images_path[i * half_batch_size : (i + 1) * half_batch_size]]
                    D_neg_batch_images = [load_unnormalized_image(path) for path in
                        D_neg_iteration_images_path[i * half_batch_size : (i + 1) * half_batch_size]]
                    # Normalize.
                    D_pos_batch_images = normalize(np.array(D_pos_batch_images)).astype(np.float32)
                    D_neg_batch_images = normalize(np.array(D_neg_batch_images)).astype(np.float32)

                    sess.run(D_optimizer, 
                             feed_dict = {D_pos_images: D_pos_batch_images,
                                          D_neg_images: D_neg_batch_images})
                    if (nc == k - 1):
                        # Positive samples' loss after training in current iteration.
                        # It will be used as an early stopping threshold when we generate pseudo-negative samples
                        D_batch_pos_loss = sess.run(D_pos_loss, 
                         feed_dict = {D_pos_images: D_pos_batch_images})
                        if (D_batch_pos_loss < min_D_batch_pos_loss):
                            min_D_batch_pos_loss = D_batch_pos_loss
                        if (D_batch_pos_loss > max_D_batch_pos_loss):
                            max_D_batch_pos_loss = D_batch_pos_loss
                # Discriminator loss after training in current iteration.
                D_last_batch_loss = sess.run(D_loss, 
                     feed_dict = {D_pos_images: D_pos_batch_images,
                                  D_neg_images: D_neg_batch_images})
                
                log(log_file_path, 
                    "Discriminator: Cascade {0}, iteration {1}, Critic {2}, time {3}, D_loss {4}, D_pos_loss {5}, {6}".format(
                    cascade, iteration, nc, time.time() - start_time, D_last_batch_loss, min_D_batch_pos_loss, max_D_batch_pos_loss))
        
            # Save last batch images in discriminator training.
            D_intermediate_image_path = os.path.join(intermediate_image_root,
                'D_cascade_{0}_iteration_{1}.png').format(cascade, iteration)
            save_unnormalized_images(images = unnormalize(np.concatenate((D_pos_batch_images,                                                                           D_neg_batch_images), axis=0)), 
                                     size = (sqrt_batch_size, sqrt_batch_size), 
                                     path = D_intermediate_image_path)
        
            ######################################################################
            # Training stage 3.1.3: Initialize pseudo-negatives.
            ######################################################################

            # Load path of negative images in last cascade and shuffle.
            S_neg_last_cascade_images_path = shuffle(S_neg_last_cascade_images_path)
            # Attention again, the last cascade here does not mean all negative images
            # produced in last cascade, but only negative images in last iteration of
            # last cascade.

            # Number of negative images to be generated in current iteration.
            if iteration == iterations_per_cascade - 1:
                # Generate more in last iteration of cascade.
                S_neg_iteration_images_count = 10000
            else:
                S_neg_iteration_images_count = 1000
            
            S_neg_current_iteration_images_path =                 [os.path.join(neg_image_root, 'cascade_{0}_iteration_{1}_count_{2}.png').format(
                    cascade, iteration + 1, i) for i in xrange(
                        S_neg_iteration_images_count)]
            
            log(log_file_path,
                  ("Sampler: Cascade {0}, iteration {1}, " + 
                   "current iteration neg {2}").format(
                   cascade, iteration, 
                   S_neg_iteration_images_count))
                  
            # Save early-stopping threshold in the model
            if iteration == iterations_per_cascade - 1:
                sess.run(D_pos_loss_min_op, {D_pos_loss_min_placeholder: 
                       min_D_batch_pos_loss})
                sess.run(D_pos_loss_max_op, {D_pos_loss_max_placeholder: 
                       max_D_batch_pos_loss})

            ######################################################################
            # Training stage 3.1.3: Sample pseudo-negatives.
            ######################################################################
            # Count of batch  in current iteration. 
            S_iteration_count_of_batch = S_neg_iteration_images_count // batch_size
            for i in xrange(S_iteration_count_of_batch):
                # Initializer the image in sampler. However, it is strange because
                # we will feed the S_images later. It is only needed if we want to
                # generate images from noise. So we ignore it at first.
                sess.run(S_initializer_op)
                sess.run(S_global_step.initializer)
                
                # Load images from last cascade generated negative images.
                # We mention it again, that is, in each iteration in current cascade, 
                # we will generate images based on last cascade, but not last iteration. 
                # It is quite a strange strategy.
                S_neg_batch_images = [load_unnormalized_image(path) for path in
                    S_neg_last_cascade_images_path[i * batch_size : 
                                                   (i + 1) * batch_size]]
                # Normalize.
                S_neg_batch_images = normalize(np.array(S_neg_batch_images)
                                              ).astype(np.float32)
                # Feed into sampler.
                sess.run(S_images_op, {S_images_placeholder: 
                                       S_neg_batch_images})

                # Generating process. We may optimize images for several times
                # to get good images. Early stopping is used here to accelerate.
                thres_ = np.random.uniform(min_D_batch_pos_loss, max_D_batch_pos_loss)
                count_of_optimizing_steps = 2000
                for j in range(count_of_optimizing_steps):
                    # Optimize.
                    sess.run(S_optimizer)
                    # Clip and re-feed to sampler.
                    sess.run(S_images_op, feed_dict = {S_images_placeholder: 
                                                       np.clip(sess.run(S_images), -1.0, 1.0)})
                    # Stop based on threshold.
                    # The threshold is based on real samples' score.
                    # Update until the WINN network thinks pseudo-negative samples are quite close to real.
                    if sess.run(S_loss) >= thres_:
                        break

                # Save intermediate negative images in sampler.
                S_neg_intermediate_images = sess.run(S_images)
                [_, height, width, channels] = S_neg_intermediate_images.shape
                for j in xrange(batch_size):
                    save_unnormalized_image(
                        image = unnormalize(S_neg_intermediate_images[j,:,:,:]),  
                        path = S_neg_current_iteration_images_path[i * batch_size + j])

                # Output information every 100 batches.
                if i % 100 == 0:
                    log(log_file_path,
                          ("Sampler: Cascade {0}, iteration {1}, batch {2}, " + 
                           "time {3}, S_loss {4}").format(
                           cascade, iteration, i, 
                           time.time() - start_time, sess.run(S_loss)))

            # After current iteration, new negative images will be added into the set of
            # negative images. Note that we keep all previous pseudo-negative images
            # to prevent the classifier forgetting what it has learned in previous stages
            
            neg_all_images_path += S_neg_current_iteration_images_path

            # Save last batch images in sampling pseudo-negatives stage.
            S_neg_intermediate_image_path = os.path.join(intermediate_image_root,
                'S_cascade_{0}_iteration_{1}.png').format(cascade, iteration)
            # In discriminator we save D_batch_images, but here we use 
            # S_intermediate_images. It is because we always use *_batch_images
            # to represent the images we put in the discriminator or sampler.
            # So G_neg_batch_images should be the "initial" images in current 
            # iteration and S_neg_intermediate_images is the generated images.
            save_unnormalized_images(images = unnormalize(S_neg_intermediate_images), 
                                     size = (sqrt_batch_size, sqrt_batch_size), 
                                     path = S_neg_intermediate_image_path)
            
        # Last cascade's generated negative images. More specifically, we only use
        # those images generated by last iteration of last cascade.
        S_neg_last_cascade_images_path = copy.deepcopy(S_neg_current_iteration_images_path)
        
        # Save the model.
        saver.save(sess, (os.path.join(model_root, 'cascade-{}.model').format(cascade)))

# Set dynamic allocation of GPU memory rather than pre-allocation.
# Also set soft placement, which means when current GPU does not exist, 
# it will change into another.
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True

# Create computation graph.
graph = tf.Graph()
with graph.as_default():
    # Set GPU number and train.
    gpu_number = 0
    with tf.device("/gpu:{0}".format(gpu_number)):    
        # Training session.
        with tf.Session(config = config) as sess:
            with tf.variable_scope("WINN", reuse = None):
                train(sess)

