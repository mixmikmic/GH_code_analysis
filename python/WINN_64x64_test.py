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

# Root directory of data directory. Customize it when using another directory.
# e.g. "./"
data_dir_root = "/mnt/cube/kwl042/church_release_candidate_3/"
# Path of data directory.
data_dir_path = os.path.join(data_dir_root, "data")

def swish(z):
    return z * tf.sigmoid(z)

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

def Normalize(name, axes, inputs):
    return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)

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
    S_vars = [var for var in trainable_vars if 'S_images' in var.name]
    
    # Checkpoint saver.
    saver = tf.train.Saver(max_to_keep = 5000)
    
    return [S_loss, S_vars, S_images, S_images_op, S_images_placeholder,
            saver, D_pos_loss_min, D_pos_loss_max,
            D_pos_loss_min_placeholder, D_pos_loss_max_placeholder,
            D_pos_loss_min_op, D_pos_loss_max_op, 
            small_noise, small_noise_op, small_noise_placeholder, big_noise]

# From https://github.com/Mazecreator/tensorflow-hints/tree/master/maximize
def maximize(optimizer, loss, **kwargs):
      return optimizer.minimize(-loss, **kwargs)

def get_optimizers(S_loss, S_vars):
    '''
    Get optimizers.
        S_loss: Sampler loss.
        S_vars: Variable to train in sampler = image.
    Return optimizer of discriminator and sampler, plus discriminator 
    learning rate, discriminator global steps and the initializer for sampler.
    '''
    
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

    return [S_optimizer, S_initializer_op, S_global_step]

def evaluate(sess):
    """
    Evaluate the WINN model.
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
    # Prepare for root directory of evaluation images.
    eval_image_root = os.path.join(data_dir_path, "evaluation")
    mkdir_if_not_exists(eval_image_root)
        
    ######################################################################
    # Training stage 1: Build network and initialize.
    ######################################################################
    log(log_file_path,
        "Training stage 1: Build network and initialize...")
    image_shape = [64, 64, 3]
    height, width, channels = image_shape
    
    # Build network.
    [S_loss, S_vars, S_images, S_images_op, S_images_placeholder,
     saver, D_pos_loss_min, D_pos_loss_max,
     D_pos_loss_min_placeholder, D_pos_loss_max_placeholder,
     D_pos_loss_min_op, D_pos_loss_max_op, 
     small_noise, small_noise_op, small_noise_placeholder, big_noise] = \
        build_network(batch_shape = [batch_size, height, width, channels])

        
    # Get optimizer.
    [S_optimizer, S_initializer_op, S_global_step] =         get_optimizers(S_loss = S_loss, S_vars = S_vars)
    
    # Show a list of global variables.
    global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='')
    log(log_file_path, "Global variables:")
    for i, var in enumerate(global_variables):
        log(log_file_path, "{0} {1}".format(i, var.name))
        
    # Initialize all variables.
    all_initializer_op = tf.global_variables_initializer()
    sess.run(all_initializer_op)
    
    # Initialize pseudo-negative images
    neg_image_root = os.path.join(data_dir_path, "evaluation")
    neg_image_path = os.path.join(eval_image_root, 'cascade_{0}_count_{1}.png')
    neg_init_images_count = 500
    neg_init_images_path = [neg_image_path.format(0, i)                             for i in range(neg_init_images_count)]
    
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
    
    eval_all_images_path = neg_init_images_path
    log(log_file_path,
        "Initial evaluation images {0}, image shape {1}".format(
        len(eval_all_images_path), image_shape))
    
    ######################################################################
    # Training stage 3: Cascades evaluation.
    ######################################################################
    log(log_file_path, "Training stage 3: Cascades evaluation...")
    
    for cascade in xrange(cascades):
        # Restore the weights.
        saver.restore(sess, (os.path.join(model_root, 'cascade-{}.model').format(cascade)))
        
        ######################################################################
        # Training stage 3.1: Prepare images for sampler evaluation.
        ######################################################################
        log(log_file_path,
              ("Sampler: Cascade {0}, " + 
               "current cascade eval {1}").format(
               cascade, 
               len(eval_all_images_path)))

        ######################################################################
        # Training stage 3.2: Sample pseudo-negatives.
        ######################################################################
        S_cascade_count_of_batch = len(eval_all_images_path) // batch_size
        for i in xrange(S_cascade_count_of_batch):
            sess.run(S_initializer_op)
            sess.run(S_global_step.initializer)
            # Load images from last cascade's sampled negative images.
            S_eval_batch_images = [load_unnormalized_image(path) for path in
                eval_all_images_path[i * batch_size : 
                                     (i + 1) * batch_size]]
            # Normalize.
            S_eval_batch_images = normalize(np.array(S_eval_batch_images)
                                           ).astype(np.float32)
            # Feed into sampler.
            sess.run(S_images_op, {S_images_placeholder: 
                                   S_eval_batch_images})

            # Sampling process. We may optimize images for several times
            # to get good images. Early stopping is used here to accelerate.
            count_of_optimizing_steps = 2000
            min_D_batch_pos_loss = sess.run(D_pos_loss_min)
            max_D_batch_pos_loss = sess.run(D_pos_loss_max)
            thres_ = np.random.uniform(min_D_batch_pos_loss, max_D_batch_pos_loss)
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

            # Save intermediate evaluation images in sampler.
            S_eval_intermediate_images = sess.run(S_images)
            [_, height, width, channels] = S_eval_intermediate_images.shape
            for j in xrange(batch_size):
                save_unnormalized_image(
                    image = unnormalize(S_eval_intermediate_images[j,:,:,:]),  
                    path = eval_all_images_path[i * batch_size + j])

            # Output information every 100 batches.
            if i % 100 == 0:
                log(log_file_path,
                      ("Sampler: Cascade {0}, batch {1}, " + 
                       "time {2}, S_loss {3}").format(
                       cascade, i, 
                       time.time() - start_time, sess.run(S_loss)))

            # Save last batch images in pseudo-negatives sampling stage.
            S_eval_intermediate_image_path = os.path.join(eval_image_root,
                'S_cascade_{0}_{1}.png').format(cascade, i)
            # In discriminator we save D_batch_images, but here we use 
            # S_eval_intermediate_images. It is because we always use *_batch_images
            # to represent the images we put in the discriminator or sampler.
            # So S_eval_batch_images should be the "initial" images in current 
            # iteration and G_eval_intermediate_images is the generated images.
            save_unnormalized_images(images = unnormalize(S_eval_intermediate_images), 
                                     size = (sqrt_batch_size, sqrt_batch_size), 
                                     path = S_eval_intermediate_image_path)

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
                evaluate(sess)



