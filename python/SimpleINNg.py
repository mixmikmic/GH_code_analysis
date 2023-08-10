__version__ = 'v1'
__author__ = 'Weijian Xu'

import os
import sys
import PIL
import time
import copy
import scipy
import sklearn
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

# Display the versions for libraries. In my environment, they are
#     Python version: 2.7.13 |Anaconda custom (64-bit)| (default, Dec 20 2016, 23:09:15) 
#     [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
#     SciPy version: 0.19.0
#     NumPy version: 1.12.1
#     TensorFlow version: 1.2.0
#     Scikit-learn version: 0.18.1
#     PIL version: 3.4.2
print('Python version: {}'.format(sys.version))
print('SciPy version: {}'.format(scipy.__version__))
print('NumPy version: {}'.format(np.__version__))
print('TensorFlow version: {}'.format(tf.__version__))
print('Scikit-learn version: {}'.format(sklearn.__version__))
print('PIL version: {}'.format(PIL.__version__))

def draw_number_on_image(image,
                         number,
                         text_format = '{:.3f}'):
    '''
    Draw a blue number on given image array.
        image:       Image array.
        number:      Number to draw.
        text_format: Format of text including the number.
    Return the modified image array.
    '''
    PIL_image = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(PIL_image)
    text = text_format.format(number)
    draw.text(xy = (0, 0), text = text, fill = (0, 0, 255, 255))
    return np.array(PIL_image)

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)

def mkdir_if_not_exists(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def get_images_path_in_directory(path):
    '''
    Get path of all images recursively in directory filtered by extension list.
        path: Path of directory contains images.
    Return path of images in selected directory.
    '''
    images_path_in_directory = []
    image_extensions = ['.png', '.jpg']
    
    for root_path, directory_names, file_names in os.walk(path):
        for file_name in file_names:
            lower_file_name = file_name.lower()
            if any(map(lambda image_extension: 
                       lower_file_name.endswith(image_extension), 
                       image_extensions)):
                images_path_in_directory.append(os.path.join(root_path, file_name))

    return images_path_in_directory

def save_unnormalized_image(image, path):
    '''
    Save one image.
        image: Unnormalized images array. The count of images 
               should match the size and the intensity values range
               from 0 to 255. Format: [height, width, channels]
        path:  Path of merged image.
    '''
    # Attention: Here we should not use the following way to save image.
    #     scipy.misc.imsave(path, image)
    # Because it automatically scale the intensity value in image
    # from [min(image), max(image)] to [0, 255]. It should be
    # the reason behind the issue reported by Kwonjoon Lee, which states 
    # the intensity value in demo in INNg paper is much near 0 or 255.
    scipy.misc.toimage(arr = image, cmin = 0, cmax = 255).save(path)

def save_unnormalized_images(images, size, path):
    '''
    Merge multiple unnormalized images into one and save it.
        images: Unnormalized images array. The count of images 
                should match the size and the intensity values range
                from 0 to 255. Format: [count, height, width, channels]
        size:   Number of images to merge. 
                Format: (vertical_count, horizontal_count).
        path:   Path of merged image.
    '''
    merged_image = merge(images, size)
    # Attention: Here we should not use the following way to save image.
    #     scipy.misc.imsave(path, merged_image)
    # Because it automatically scale the intensity value in merged_image
    # from [min(merged_image), max(merged_image)] to [0, 255]. It should be
    # the reason behind the issue reported by Kwonjoon Lee, which states 
    # the intensity value in demo in INNg paper is much near 0 or 255.
    scipy.misc.toimage(arr = merged_image, cmin = 0, cmax = 255).save(path)

def load_unnormalized_image(path):
    '''
    Load a RGB image and do not normalize. Each intensity value is from 
    0 to 255 and then it is converted into 32-bit float.
        path: Path of image file.
    Return image array.
    '''
    return scipy.misc.imread(path, mode = 'RGB').astype(np.float32)

def merge(images, size):
    '''
    Merge several images into one.
        size: Number of images to merge. 
              Format: (vertical_count, horizontal_count)
    Return merged image array.
    '''
    count, height, width, channels = images.shape
    vertical_count, horizontal_count = size
    if not (vertical_count * horizontal_count == count):
        raise ValueError("Count of images does not match size.")
        
    # Merged image looks like
    #     [ ][ ][ ]
    #     [ ][ ][ ]
    #     [ ][ ][ ]
    # when size = [3, 3].
    merged_image = np.zeros((height * vertical_count, 
                             width * horizontal_count, 
                             channels))
    for i, image in enumerate(images):
        m = i // vertical_count
        n = i % vertical_count
        merged_image[m * height : (m + 1) * height, 
                     n * width : (n + 1) * width, :] = image
    return merged_image 

def normalize(images):
    '''
    Normalize the intensity values from [0, 255] into [-1, 1].
        images: Image array to normalize. Require each intensity value
                ranges from 0 to 255.
    Return normalized image array.
    '''
    return 1.0 * np.array(images) / 255 * 2.0 - 1.0

def unnormalize(images):
    '''
    Unnormalize the intensity values from [-1, 1] to [0, 255].
        images: Image array to unnormalize. Require each intensity value 
                ranges from -1 to 1.
    Return unnormalized image array.
    '''
    return (images + 1.0) / 2.0 * 255

def gen_unnormalized_random_images(image_shape, count):
    '''
    Generate unnormalized image with random intensity values. Each intensity
    value ranges from 0 to 255.
        image_shape: Shape of an image. Format: [height, width, channels]
        count:       Number of random images to generate.
    Return array of generated random images.
    '''
    height, width, channels = image_shape
    intermediate_images = np.random.normal(loc = 0, scale = 0.3, 
                          size = [count, height, width, channels])
    intermediate_images = intermediate_images - intermediate_images.min()
    return intermediate_images / intermediate_images.max() * 255.0 

def get_image_shape(path):
    '''
    Get shape of image. Format: [height, width, channels]. In fact, all images
    are regarded as color images, thus, channels is always 3.
        path: Path of image file.
    Return array of image shape.
    '''
    image = scipy.misc.imread(path)
    [height, width, channels] = image.shape
    return [height, width, channels]

# Root directory of data directory. Customize it when using another directory.
# e.g. "./"
data_dir_root = "./"
# Path of data directory.
data_dir_path = os.path.join(data_dir_root, "data")
# Path of dataset directory.
dataset_dir_path = os.path.join(data_dir_root, "dataset")
# Path of CelebA dataset directory.
celeba_dir_path = os.path.join(dataset_dir_path, "celeba")
# Path of CelebA dataset directory for cropped images.
celeba_cropped_dir_path = os.path.join(celeba_dir_path, "cropped")

# Create a series of directories.
mkdir_if_not_exists(data_dir_path)

def conv2d(input, output_channels, scope):
    '''
    2-D convolutional layer.
        input:           Input layer.
        output_channels: Output channels.
        scope:           Variable scope.
    Return output of 2-D convolutional layer.
    '''
    
    # Pre-defined parameters.
    in_height = 5
    in_width = 5 
    in_channels = input.get_shape().as_list()[-1]
    delta_height = 2
    delta_width = 2 
    
    with tf.variable_scope(scope):
        # Variable for filter.
        filter = tf.get_variable(
            name = 'filter', 
            shape = [in_height, in_width, in_channels, output_channels],
            initializer = tf.truncated_normal_initializer(stddev = 0.02),
            regularizer = tf.contrib.layers.l2_regularizer(scale = 0.0002)
        )
        # Do convolution.
        conv = tf.nn.conv2d(input, 
                            filter, 
                            strides = [1, delta_height, delta_width, 1], 
                            padding = 'SAME')
        # Variable for bias.
        bias = tf.get_variable(name = 'bias', 
                               shape = [output_channels], 
                               initializer = tf.constant_initializer(0.0))
        # Add bias.
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

        return conv

def lrelu(input):
    '''
    Leaky ReLU layer.
        input: Input layer.
    Return output of leaky ReLU layer.
    '''
    
    # Pre-defined leak coefficient. 
    leak = 0.2
    
    # Do leaky ReLU and return.
    return tf.maximum(input, leak * input)

def linear(input, output_size, scope):
    '''
    Linear layer.
        input:       Input layer.
        output_size: Output size.
        scope:       Variable scope.
    Return output of linear layer.
    '''
    
    # Only consider input is 2-D, that is, shape is [batch_size, input_size].
    shape = input.get_shape().as_list()
    batch_size, input_size = shape
    
    with tf.variable_scope(scope):
        # Variable of matrix.
        matrix = tf.get_variable(
            name = "matrix", 
            shape = [input_size, output_size], 
            dtype = tf.float32,
            initializer = tf.random_normal_initializer(stddev = 0.02),
            regularizer = tf.contrib.layers.l2_regularizer(0.0002)
        )
        
        # Variable of bias.
        bias = tf.get_variable(name = "bias", 
                               shape = [output_size],
                               initializer = tf.constant_initializer(0.0))

        # Do multiplication and return.
        return tf.matmul(input, matrix) + bias

def batch_norm(input, is_training, reuse, scope):
    '''
    Batch normalization layer.
        input:       Input layer.
        is_training: True in training, False in evaluation.
        reuse:       Reuse weights or not.
        scope:       Variable scope.
    Return output of batch normalization layer.
    '''
    return tf.contrib.layers.batch_norm(input,
                                        decay = 0.9, 
                                        updates_collections = None,
                                        epsilon = 1e-5,
                                        scale = True,
                                        is_training = is_training,
                                        reuse = reuse,
                                        scope = scope)

def discriminator(images):
    '''
    Build discriminator.
        images: Training images. Shape is [batch size, height, width, channels]
    Return the predicted outputs of sigmoid and its logits.
    '''
    
    # Consider scope layers. Because we build discriminator at first, 
    # reuse shoule be False here to create variables. Later when we build
    # generator, reuse should be True as variables have been created.
    with tf.variable_scope("layers", reuse = False):

        # Pre-defined dimension of filters in first layer.
        filter_dim = 64

        # Fetch batch size from images.
        batch_size = images.get_shape().as_list()[0]

        # Hidden layer 0.
        h0 = lrelu(conv2d(images, filter_dim, scope = 'h0_conv'))
        
        # Hidden layer 1.
        h1_conv = conv2d(h0, filter_dim * 2, scope = 'h1_conv')
        h1_bn = batch_norm(h1_conv, is_training = True,
                           reuse = False, scope = 'h1_bn')
        h1 = lrelu(h1_bn)

        # Hidden layer 2.
        h2_conv = conv2d(h1, filter_dim * 4, scope = 'h2_conv')
        h2_bn = batch_norm(h2_conv, is_training = True,
                           reuse = False, scope = 'h2_bn')
        h2 = lrelu(h2_bn)

        # Hidden layer 3.
        h3_conv = conv2d(h2, filter_dim * 8, scope = 'h3_conv')
        h3_bn = batch_norm(h3_conv, is_training = True, 
                           reuse = False, scope = 'h3_bn')
        h3 = lrelu(h3_bn)

        # Hidden layer 4.
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 
                    output_size = 1, scope = 'h4_linear')

        return tf.nn.sigmoid(h4), h4

def generator(images):  
   '''
   Build generator.
       images: Training images. Shape is [batch size, height, width, channels]
   Return the predicted outputs of sigmoid and its logits.
   '''
   
   # Consider scope layers. Always reuse the variables because those variables
   # have been created in discriminator.
   with tf.variable_scope("layers", reuse = True):
       
       # Pre-defined dimension of filters in first layer.
       filter_dim = 64

       # Fetch batch size from images.
       batch_size = images.get_shape().as_list()[0]

       # Hidden layer 0.
       h0 = lrelu(conv2d(images, filter_dim, scope = 'h0_conv'))
   
       # Hidden layer 1.
       h1_conv = conv2d(h0, filter_dim * 2, scope = 'h1_conv')
       h1_bn = batch_norm(h1_conv, is_training = False,
                          reuse = True, scope = 'h1_bn')
       h1 = lrelu(h1_bn)

       # Hidden layer 2.
       h2_conv = conv2d(h1, filter_dim * 4, scope = 'h2_conv')
       h2_bn = batch_norm(h2_conv, is_training = False,
                          reuse = True, scope = 'h2_bn')
       h2 = lrelu(h2_bn)

       # Hidden layer 3.
       h3_conv = conv2d(h2, filter_dim * 8, scope = 'h3_conv')
       h3_bn = batch_norm(h3_conv, is_training = False, 
                          reuse = True, scope = 'h3_bn')
       h3 = lrelu(h3_bn)

       # Hidden layer 4.
       h4 = linear(tf.reshape(h3, [batch_size, -1]), 
                   output_size = 1, scope = 'h4_linear')

       return tf.nn.sigmoid(h4), h4

def build_network(batch_shape):
    '''
    Build an INNg model as network.
        batch_shape: Shape of a mini-batch in discriminating and generating.
                     The format is [batch size, height, width, channels].
    Return loss, trainable variables, labels and images in discriminator and 
    generator, plus checkpoint saver. 
    '''

    # Get details of batch shape.
    [batch_size, height, width, channels] = batch_shape
    
    # Placeholder of images and labels.
    D_images = tf.placeholder(dtype = tf.float32, 
                              shape = [batch_size, height, width, channels], 
                              name = 'D_images')
    D_labels = tf.placeholder(dtype = tf.float32, 
                              shape = [batch_size, 1], 
                              name = 'D_labels')

    # Variable, placeholder and assign operator for multiple generated images.
    G_images = tf.Variable(
        # Use uniform distribution Unif(-1, 1) to initialize.
        np.random.uniform(low = -1.0, 
                          high = 1.0, 
                          size = [batch_size, height, width, channels]
        ).astype('float32'), 
        name='G_images'
    )
    G_images_placeholder = tf.placeholder(dtype = G_images.dtype, 
                                          shape = G_images.get_shape())
    G_images_op = G_images.assign(G_images_placeholder)

    # Build discriminator.
    # D is sigmoid(D_logits).
    D, D_logits = discriminator(D_images)
    D_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits = D_logits, 
            labels = D_labels
        )
    )

    # Build generator.
    # G is sigmoid(G_logits).
    G, G_logits = generator(G_images)
    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits = G_logits, 
            # All labels point to positive examples, which is similar as GAN.
            # 
            # Thus, minimize G_loss
            #    => minimize -log(sigmoid(G_logits))
            #    => maximize G_logits
            #    => maximize g_t(x) in the paper
            #    => maximize ln g_t(x) in the paper
            labels = tf.ones_like(G_logits) * 1.0
        )
    )

    # Variables to train.
    trainable_vars = tf.trainable_variables()
    D_vars = [var for var in trainable_vars if 'layers' in var.name]
    G_vars = [var for var in trainable_vars if 'G_images' in var.name]
    
    # Checkpoint saver.
    saver = tf.train.Saver(max_to_keep = 5000)
    
    return [D_loss, G_loss, D_vars, G_vars, 
            D_images, G_images, G_images_op, G_images_placeholder,
            D_labels, saver]

def get_optimizers(D_loss, G_loss, D_vars, G_vars):
    '''
    Get optimizers.
        D_loss: Discriminator loss.
        G_loss: Generator loss.
        D_vars: Variables to train in discriminator.
        G_vars: Variables to train in generator.
    Return optimizer of discriminator and generator, plus discriminator 
    learning rate, discriminator global steps and initializer for generator.
    '''
    
    # Scope of discriminator optimizer.
    with tf.variable_scope('D_optimizer'):
        # Learning rate.
        D_learning_rate = 0.01
        D_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate = D_learning_rate).minimize(
                loss = D_loss, var_list = D_vars)
        
    # Scope of generator optimizer.
    with tf.variable_scope('G_optimizer'):
        G_adam = tf.train.AdamOptimizer(learning_rate = 0.02, beta1 = 0.5)
        G_optimizer = G_adam.minimize(loss = G_loss, var_list = G_vars)
    
    # Variables of generator optimizer and initializer operator of that.
    G_optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                         scope = 'INNg/G_optimizer')
    G_optimizer_initializer_op = tf.variables_initializer(G_optimizer_vars)
    
    return [D_optimizer, G_optimizer, 
            D_learning_rate,
            G_optimizer_initializer_op]

def train(sess):
    """
    Train the INNg model.
        sess: Session.
    """

    # Set timer.
    start_time = time.time()
    # Batch size. It should be a squared number.
    batch_size = 100
    half_batch_size = batch_size // 2
    sqrt_batch_size = int(np.sqrt(batch_size))
    # How many cascades in INNg.
    cascades = 100
    # How many iterations for each cascade classifier.
    iterations_per_cascade = 100
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
    # Training stage 1: Load positive and negative images.
    ######################################################################
    log(log_file_path,
        "Training stage 1: Load positive and negative images...")
    
    def gen_neg_init_images(neg_image_root, image_shape):
        '''
        Generate negative initial images.
            data_dir_path: Data directory to contain negative images directory.
            image_shape:   Shape of image. Format: [height, width, channels]
        Return array of generated initial images path and their root directory.
        '''
        # In fact, the name of image has format 
        #     {cascade}_{next iteration}_{i}.png
        # where cascade means current cascade model, next iteration means
        # next iteration of generator and discriminator training, and i means
        # the index of images.
        neg_image_path = os.path.join(neg_image_root, 'cascade_{0}_iteration_{1}_count_{2}.png')
        neg_init_images_count = 10000
        neg_init_images_path = [neg_image_path.format(0, 0, i)                                 for i in range(neg_init_images_count)]
       
        # Generate random images as negatives and save them.
        for i, neg_init_image_path in enumerate(neg_init_images_path):
            # Attention: Though it is called neg_image here, it has 4 dimensions,
            #            that is, [1, height, width, channels], which is not a
            #            pure single image, which is [height, width, channels].
            #            So we still use save_unnormalized_images here instead of 
            #            save_unnormalized_image.
            neg_image = gen_unnormalized_random_images(
                image_shape = image_shape, count = 1)
            save_unnormalized_images(images = neg_image, 
                                     size = (1, 1), path = neg_init_image_path)
            
        return neg_init_images_path
        
    # Path of all positive images and negative images. 
    # The following celeba_cropped_dir_path can be replaced. The image shape of
    # positive and negative images are the same.
    pos_all_images_path = get_images_path_in_directory(celeba_cropped_dir_path)
    image_shape = get_image_shape(pos_all_images_path[0])
    neg_all_images_path = gen_neg_init_images(
        neg_image_root = neg_image_root, 
        image_shape = image_shape)
    log(log_file_path,
        "Positive images {0}, negative images {1}, image shape {2}".format(
        len(pos_all_images_path), len(neg_all_images_path), image_shape))

    ######################################################################
    # Training stage 2: Build network and initialize.
    ######################################################################
    log(log_file_path,
        "Training stage 2: Build network and initialize...")
    height, width, channels = image_shape
    
    # Build network.
    [D_loss, G_loss, D_vars, G_vars, 
     D_images, G_images, G_images_op, G_images_placeholder,
     D_labels, saver] = \
        build_network(batch_shape = [batch_size, height, width, channels])
        
    # Get optimizer.
    [D_optimizer, G_optimizer, 
     D_learning_rate,
     G_optimizer_initializer_op] = \
        get_optimizers(D_loss = D_loss, G_loss = G_loss, 
                       D_vars = D_vars, G_vars = G_vars)
    
    # Show a list of global variables.
    global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='')
    log(log_file_path, "Global variables:")
    for i, var in enumerate(global_variables):
        log(log_file_path, "{0} {1}".format(i, var.name))
        
    # Initialize all variables.
    all_initializer_op = tf.global_variables_initializer()
    sess.run(all_initializer_op)
    
    ######################################################################
    # Training stage 3: Cascades training.
    ######################################################################
    log(log_file_path, "Training stage 3: Cascades training...")
    
    # INNg single: Only one cascade with multiple iterations.
    # INNg cascade: Multiple cascades, each cascade with one iteration.
    # (model_cascade.py in Long Jin's code, but I have not checked it)
    # INNg compact: Multiple cascades, each cascade with multiple iterations.
    # (model_few.py in Long Jin's code)
    # Here we use INNg compact model. The definition of cascade and iteration
    # can be found below.
    
    # One cascade means one new model.
    # p <- [cascade n] <- [cascade n-1] <- ... <- [cascade 0] <- p_r 
    # where p_r means Uniform or Gaussian reference distribution 
    # and p means estimated negative distribution in paper.
    # One cascade training may contain multiple iterations.
    
    # Prepare for the initial images to feed the generator. In fact, it is 
    # because we always use negative images in last cascade as the "initial"
    # images to feed generator in all iterations of current cascade.
    G_neg_last_cascade_images_path = copy.deepcopy(neg_all_images_path)
    
    for cascade in xrange(cascades):
        ######################################################################
        # Training stage 3.1: Iterations training.
        ######################################################################
        # One iteration means one time of discriminator training and one time
        # of generator training. One iteration training may contain multiple
        # batches for discriminator and generator training.
        for iteration in xrange(iterations_per_cascade):
            ######################################################################
            # Training stage 3.1.1: Prepare images and labels for discriminator
            # training.
            ######################################################################
            # Count of positive images to train in current iteration.
            # Still, it is a strange strategy from Long Jin's code.
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
            # has different behaviors on discriminator and generator.
            # 1) Discriminator.
            #     We draw positive images from all positive images and the same
            #     number of negative images from all negative images. Every 
            #     iteration of generator will add newly generated negative images
            #     into all negative images.
            # 2) Generator.
            #     We draw "initial" negative images in every iterations in current
            #     cascade from part of generated negative images in last cascade.
            #     More specificially, the part is the last iteration of last cascade.
            #     When current cascade ends, the generated negative images will be
            #     used in next cascade.
            D_neg_iteration_images_count = D_pos_iteration_images_count
            D_neg_iteration_images_path = np.random.choice(
                neg_all_images_path,
                D_pos_iteration_images_count, 
                replace = True).tolist()
                
            # Images and labels in discriminator training in current iteration.
            D_iteration_images_path = D_pos_iteration_images_path +                                       D_neg_iteration_images_path
            # Labels need to be reshaped into 2-D in order to match the shape
            # of output of last linear layer in discriminator.
            D_iteration_labels = np.array(
                [1.0] * D_pos_iteration_images_count + \
                [0.0] * D_neg_iteration_images_count
            ).reshape(-1, 1)
            # Shuffle current iteration images with labels.
            D_iteration_images_path, D_iteration_labels =                 shuffle(D_iteration_images_path, D_iteration_labels)
            
            log(log_file_path,
                   ("Discriminator: Cascade {0}, iteration {1}, " + 
                   "all pos {2}, all neg {3}, " + 
                   "current iteration {4} (pos {5}, neg {6}), " + 
                   "learning rate {7}").format(
                       cascade, iteration, 
                       len(pos_all_images_path), len(neg_all_images_path), 
                       len(D_iteration_images_path), 
                       D_pos_iteration_images_count, D_neg_iteration_images_count, 
                       D_learning_rate
                   ))
            
            ######################################################################
            # Training stage 3.1.2: Train the discriminator.
            ######################################################################
            # Count of batch in discriminator training in current iteration. 
            D_iteration_count_of_batch = len(D_iteration_images_path) // batch_size
            for i in xrange(D_iteration_count_of_batch):
                # Load images for this batch in discriminator.
                D_batch_images = [load_unnormalized_image(path) for path in
                    D_iteration_images_path[i * batch_size : (i + 1) * batch_size]]
                # Normalize.
                D_batch_images = normalize(np.array(D_batch_images)).astype(np.float32)
                # Read labels for this batch in discriminator.
                D_batch_labels = D_iteration_labels[i * batch_size : (i + 1) * batch_size]
                # Currently the decay is disabled according to Long Jin's reply.
                sess.run(D_optimizer, 
                         feed_dict = {D_images: D_batch_images, 
                                      D_labels: D_batch_labels})

            # Discriminator loss after training in current iteration.
            D_last_batch_loss = sess.run(D_loss, 
                 feed_dict = {D_images: D_batch_images, 
                              D_labels: D_batch_labels})
    
            log(log_file_path, 
                "Discriminator: Cascade {0}, iteration {1}, time {2}, D_loss {3}".format(
                cascade, iteration, time.time() - start_time, D_last_batch_loss))
        
            # Save last batch images in discriminator training.
            D_intermediate_image_path = os.path.join(intermediate_image_root,
                'D_cascade_{0}_iteration_{1}.png').format(cascade, iteration)
            save_unnormalized_images(images = unnormalize(D_batch_images), 
                                     size = (sqrt_batch_size, sqrt_batch_size), 
                                     path = D_intermediate_image_path)
        
            ######################################################################
            # Training stage 3.1.3: Prepare images for generator training.
            ######################################################################

            # Load path of negative images in last cascade and shuffle.
            G_neg_last_cascade_images_path = shuffle(G_neg_last_cascade_images_path)
            # Attention again, the last cascade here does not mean all negative images
            # produced in last cascade, but only negative images in last iteration of
            # last cascade.

            # Count of negative images generated in current iteration.
            # Strange strategy from Long Jin's code.
            if iteration == iterations_per_cascade - 1:
                # Generate more in last iteration of cascade.
                G_neg_iteration_images_count = 10000
            else:
                G_neg_iteration_images_count = 1000
            
            G_neg_current_iteration_images_path =                 [os.path.join(neg_image_root, 'cascade_{0}_iteration_{1}_count_{2}.png').format(
                    cascade, iteration + 1, i) for i in xrange(
                        G_neg_iteration_images_count)]
            
            log(log_file_path,
                  ("Generator: Cascade {0}, iteration {1}, " + 
                   "current iteration neg {2}").format(
                   cascade, iteration, 
                   G_neg_iteration_images_count))
                  
            ######################################################################
            # Training stage 3.1.3: Train the generator.
            ######################################################################
            # Count of batch in generator training in current iteration. 
            G_iteration_count_of_batch = G_neg_iteration_images_count // batch_size
            for i in xrange(G_iteration_count_of_batch):
                # Initialize the Adam optimizer every iteration.
                sess.run(G_optimizer_initializer_op)

                # Load images from last cascade generated negative images.
                # We mention it again, that is, in each iteration in current cascade, 
                # we will generate images based on last cascade, but not last iteration. 
                # It is quite a strange strategy.
                G_neg_batch_images = [load_unnormalized_image(path) for path in
                    G_neg_last_cascade_images_path[i * batch_size : 
                                                   (i + 1) * batch_size]]
                # Normalize.
                G_neg_batch_images = normalize(np.array(G_neg_batch_images)
                                              ).astype(np.float32)
                # Feed into generator.
                sess.run(G_images_op, {G_images_placeholder: 
                                       G_neg_batch_images})

                # Generating process. We may optimize images for several times
                # to get good images. Early stopping is used here to accelerate.
                count_of_optimizing_steps = 2000
                for j in range(count_of_optimizing_steps):
                    # Optimize.
                    sess.run(G_optimizer)
                    # Clip and re-feed to generator.
                    sess.run(G_images_op, feed_dict = {G_images_placeholder: 
                                                       np.clip(sess.run(G_images), -1.0, 1.0)})
                    # Early stopping based on threshold.
                    # The threshold is based on decision boundary of cross entropy.
                    #     -ln(sigmoid(0)) = -ln(0.5) = ln 2 = 0.693
                    # The boundary means it is more likely for the discriminator
                    # to consider the images as postive.
                    if sess.run(G_loss) <= 0.693:
                        break

                # Save intermediate negative images in generator.
                G_neg_intermediate_images = sess.run(G_images)
                [_, height, width, channels] = G_neg_intermediate_images.shape
                for j in xrange(batch_size):
                    save_unnormalized_image(
                        image = unnormalize(G_neg_intermediate_images[j,:,:,:]),  
                        path = G_neg_current_iteration_images_path[i * batch_size + j])

                # Output information every 100 batches.
                if i % 100 == 0:
                    log(log_file_path,
                          ("Generator: Cascade {0}, iteration {1}, batch {2}, " + 
                           "time {3}, G_loss {4}").format(
                           cascade, iteration, i, 
                           time.time() - start_time, sess.run(G_loss)))

            # After current iteration, new negative images will be added into all
            # negative images. In fact, it is strange because we only add 1,000 images
            # every iteration except last iteration, in which we add 10,000 images.
            
            neg_all_images_path += G_neg_current_iteration_images_path

            # Save last batch images in generator training.
            G_neg_intermediate_image_path = os.path.join(intermediate_image_root,
                'G_cascade_{0}_iteration_{1}.png').format(cascade, iteration)
            # In discriminator we save D_batch_images, but here we use 
            # G_intermediate_images. It is because we always use *_batch_images
            # to represent the images we put in the discriminator or generator.
            # So G_neg_batch_images should be the "initial" images in current 
            # iteration and G_neg_intermediate_images is the generated images.
            save_unnormalized_images(images = unnormalize(G_neg_intermediate_images), 
                                     size = (sqrt_batch_size, sqrt_batch_size), 
                                     path = G_neg_intermediate_image_path)
            
        # Last cascade's generated negative images. More specifically, we only use
        # those images generated by last iteration of last cascade.
        G_neg_last_cascade_images_path = copy.deepcopy(G_neg_current_iteration_images_path)
        
        # Save the model.
        saver.save(sess, (os.path.join(model_root, 'cascade-{}.model').format(cascade)))

# Set dynamic allocation of GPU memory rather than pre-allocation.
# Also set soft placement, which means when current GPU does not exist, 
# it will change into another.
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
# Set GPU number.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Create computation graph.
graph = tf.Graph()
with graph.as_default():    
    # Training session.
    with tf.Session(config = config) as sess:
        with tf.variable_scope("INNg", reuse = None):
            train(sess)

