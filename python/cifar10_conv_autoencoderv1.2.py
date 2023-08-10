import numpy as np
import pickle
import os
import urllib
import tarfile
import zipfile
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import time

def _print_download_progress(count, block_size, total_size):
    """
    Helper function to visualize the download in progress
    Used as a call-back function
    """
    
    # percentage completion
    pct_complete = float(count * block_size) / total_size
    
    # Status message. \r means that the line should overwrite itself
    msg = "\r - Download progress : {0:.1%}".format(pct_complete)
    
    # print it
    sys.stdout.write(msg)
    sys.stdout.flush()

def maybe_download_and_extract(url, download_dir):
    """
    Download and extract the data if it doesn't already exist
    """
    # Filename for saving the downloaded file
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)
    
    # Check if the file already exists. If it already exists then we assume it 
    # has already been extracted. Else we need to download and extract
    
    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        
        # download the file from the internet
        file_path, _ = urllib.urlretrieve(url, file_path, _print_download_progress)
        
        print()
        print('Download finished. Extracting files.')
        
        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
        print('Done.')
    else:
        print('Data has apparently already been downloaded and unpacked')

# Directory where you want to download and save the data-set
data_path = 'data_cifar10/'

# Url for the data-set on the internet
data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# width and height of the image
img_size = 32

# number of channels in the image
num_channels = 3

# length of the image when flattened into a 1-d array
img_size_flat = img_size * img_size * num_channels

# number of classes
num_classes = 10

# number of files for the training set
_num_files_train = 5

# number of images for each batch-file in the training set
_images_per_file = 10000

# total number of images in the training set
_num_images_train = _num_files_train * _images_per_file

def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set
    """
    return os.path.join(data_path, "cifar-10-batches-py/", filename)

    
    
def _unpickle(filename):
    """
    Unpickle the given file and return its data
    """
    file_path = _get_file_path(filename)
    print("Loading data:" + file_path)
    
    with open(file_path, mode='rb') as file:
        data = pickle.load(file)
    
    return data


    
def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and 
    return a 4-dim array with shape : [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0
    """
    # Convert the raw images from the data-files to floating-points
    raw_float = np.array(raw, dtype=float) / 255.0
    
    # Reshape the array into 4-dim
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    
    # Reorder the indices of the array
    images = images.transpose([0, 2, 3, 1])
    
    return images



    
def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images
    """
    
    # Load the pickled data-file
    data = _unpickle(filename)
    
    # Get the raw images
    raw_images = data[b'data']
    
    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)
    
    return images, cls


def load_class_names():
    """
    Load the names for the classes in the CIFAR-10 data-set.
    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """

    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode("utf-8") for x in raw]

    return names


def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-hot encoded class-labels from an array of integers
    
    For example, if class_number=2 and num_classes=4 then the 
    one-hot encoded label is the float array: [0. 0. 1. 0.]
    """
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1
    
    return np.eye(num_classes, dtype=float)[class_numbers]
    
def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set
    The data-set is split into 5 data-files which are merged here
    Returns the images, class-numbers and one-hot encoded class-labels
    """
    
    # Pre-allocate the arrays for the images and class-numbers for efficiency
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)
    
    # Begin-index for the current batch
    begin = 0
    
    # For each data-file
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))
        
        # number of images in this batch
        num_images = len(images_batch)
        
        # end index of the current batch
        end = begin + num_images
        
        # Store the images into the array
        images[begin:end, :] = images_batch
        
        # Store the class-numbers into the array
        cls[begin:end] = cls_batch
        
        # the begin index for the next batch is the current end-index
        begin = end
        
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
    """
    Load all the test data for the dataset
    
    Returns the images, class-numbers and one-hot encoded class-labels
    """
    images, cls = _load_data(filename="test_batch")
    
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

# Load the class-names
class_names = load_class_names()
class_names

# Load the training-set.
# This returns the images, the class-numbers as integers, 
# and the class-numbers as One-Hot encoded arrays called labels

images_train, cls_train, labels_train = load_training_data()
images_train.shape

images_train.nbytes

# Load the test set

images_test, cls_test, labels_test = load_test_data()

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

# images are 32x32 pixels but we will crop the images to 24x24 pixels
img_size_cropped = 24

# Helper function to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image

def plot_images(images, cls_true, cls_pred=None, smooth=True):
    
    assert len(images) == len(cls_true) == 9
    
    # create figure with sub-plots
    fig, axes = plt.subplots(3, 3)
    
    # adjust vertical spacing if we need to print ensemble and best-net
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)
    
    for i, ax in enumerate(axes.flat):
        # Interpolation type
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
        
        # Plot image
        ax.imshow(images[i, :, :, :], interpolation=interpolation)
        
        # Name of the true class
        cls_true_name = class_names[cls_true[i]]
        
        # Show the true and predicted classes
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # name of the predicted class
            cls_pred_name = class_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        
        # show the classes as the label on the x-axis
        ax.set_xlabel(xlabel)
        
        # remove the ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# Get the first images from the test-set
images = images_test[0:9]

# Get the true classes from those images
cls_true = cls_test[0:9]

# Plot the images and labels using the helper function
plot_images(images=images, cls_true=cls_true, smooth=False)

# the pixelated images above are what the neural net will get as an input. the images must be a bit easier for the
# human eye to recognize

plot_images(images=images, cls_true=cls_true, smooth=True)

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
img_batch_size = tf.shape(x)[0]

def pre_process_image(image, training):
    # This function takes a single image as an input
    # and a boolean whether to build the training or testing graph
    
    if training:
        # For training, add the following to the tensor graph
        
        # randomly crop the input image
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
        
        # Randomly flip the image horizontally
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0., upper=2.)
        
        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
        
    else:
        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)
    return image

def pre_process(images, training):
    # Use TF to loop over all the input images and call
    # the function above which takes a single image as input
    
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)
    
    return images

# To plot the distorted images. Only graph is created. Executed later

distorted_images = pre_process(images=x, training=True)

# Create the model

def main_network(X, training=True, scope='dummy'):
    
    images = pre_process(X, training=training)
    print(images.shape)
    
    with tf.variable_scope(scope) as sc:
    
        # convolution 1
        wc1 = tf.get_variable('wc1', 
                              shape=(3, 3, 3, 32), 
                              initializer=tf.truncated_normal_initializer()) # weights
        conv1 = tf.nn.conv2d(images, wc1, strides=[1, 1, 1, 1], padding='SAME')
        bc1 = tf.get_variable('bc1', shape=(32,), initializer=tf.constant_initializer(0)) # biases
        conv1 = tf.nn.sigmoid(tf.add(conv1, bc1))
        # maxpooling
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(conv1.shape)
    
        # convolution 2
        wc2 = tf.get_variable('wc2', 
                              shape=(3, 3, 32, 64), 
                              initializer=tf.truncated_normal_initializer()) # weights
        conv2 = tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME')
        bc2 = tf.get_variable('bc2', shape=(64,), initializer=tf.constant_initializer(0)) # biases
        conv2 = tf.nn.sigmoid(tf.add(conv2, bc2))
        # maxpooling
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(conv2.shape)
        
        # convolution 3
        wc3 = tf.get_variable('wc3', 
                              shape=(3, 3, 64, 64), 
                              initializer=tf.truncated_normal_initializer()) # weights
        conv3 = tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME')
        bc3 = tf.get_variable('bc3', shape=(64,), initializer=tf.constant_initializer(0)) # biases
        conv3 = tf.nn.sigmoid(tf.add(conv3, bc3))
        # maxpooling
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(conv3.shape)
    
        # reshape the conv2 into a fully connected layer input
        wfc1 = tf.get_variable('wfc1', 
                               shape=(3*3*64, 400), 
                               initializer=tf.contrib.layers.xavier_initializer()) # weights
        bf1 = tf.get_variable('bf1', shape=(400,), initializer=tf.constant_initializer(0)) # biases
        fc1 = tf.reshape(conv3, shape=[-1, wfc1.get_shape().as_list()[0]])
    
        # fully connected layer 1
        fc1 = tf.nn.sigmoid(tf.add(tf.matmul(fc1, wfc1), bf1))
        print(fc1.shape)
    
        # fully connected layer 2 or the middle representation
        wfc2 = tf.get_variable('wfc2', 
                               shape=(400, 256), 
                               initializer=tf.contrib.layers.xavier_initializer()) # weights
        bf2 = tf.get_variable('bf2', shape=(256,), initializer=tf.constant_initializer(0)) # biases
        fc2 = tf.nn.sigmoid(tf.add(tf.matmul(fc1, wfc2), bf2))
        print(fc2.shape)
        
        encoded_output = fc2
        print('Input encoded!')
    
        # deconvolution fully connected layer 1
        wdf1 = tf.get_variable('wdf1', 
                               shape=(256, 400), 
                               initializer=tf.contrib.layers.xavier_initializer()) # weights
        bdf1 = tf.get_variable('bdf1', shape=(400,), initializer=tf.constant_initializer(0)) # biases
        dfc1 = tf.add(tf.matmul(encoded_output, wdf1), bdf1)
        dfc1 = tf.nn.sigmoid(dfc1)
        print(dfc1.shape)
        
        # deconvolution, fully connected layer 2
        wdf2 = tf.get_variable('wdf2',
                               shape=(400, 3*3*64),
                               initializer=tf.contrib.layers.xavier_initializer()) # weights
        bdf2 = tf.get_variable('bdf2', shape=(3*3*64,), initializer=tf.constant_initializer(0)) # biases
        dfc2 = tf.add(tf.matmul(dfc1, wdf2), bdf2)
        dfc2 = tf.nn.sigmoid(dfc2)
        
        # deconvolution 1
        wd1 = tf.get_variable('wd1', 
                              shape=(3, 3, 64, 64), 
                              initializer=tf.truncated_normal_initializer) # weights
        bd1 = tf.get_variable('bd1', shape=(64,), initializer=tf.constant_initializer(0)) # biases
        # reshape the output from the fully connected layer 2
        dfc2 = tf.reshape(dfc2, shape=[-1, 3, 3, 64])
        deconv1 = tf.nn.conv2d_transpose(dfc2, 
                                         wd1, 
                                         output_shape=[img_batch_size, 6, 6, 64], 
                                         strides=[1, 2, 2, 1], 
                                         padding='SAME')
        deconv1 = tf.nn.sigmoid(tf.add(deconv1, bd1))
        print(deconv1.shape)
        
        # deconvolution 2
        wd2 = tf.get_variable('wd2', 
                              shape=(3, 3, 32, 64), 
                              initializer=tf.truncated_normal_initializer) # weights
        bd2 = tf.get_variable('bd2', shape=(32,), initializer=tf.constant_initializer(0)) # biases
        deconv2 = tf.nn.conv2d_transpose(deconv1, 
                                         wd2, 
                                         output_shape=[img_batch_size, 12, 12, 32], 
                                         strides=[1, 2, 2, 1], 
                                         padding='SAME')
        deconv2 = tf.nn.sigmoid(tf.add(deconv2, bd2))
        print(deconv2.shape)
        
        # deconvolution 3
        wd3 = tf.get_variable('wd3', 
                              shape=(3, 3, 3, 32), 
                              initializer=tf.truncated_normal_initializer()) # weights
        bd3 = tf.get_variable('bd3', shape=(3,), initializer=tf.constant_initializer(0)) # biases
        deconv3 = tf.nn.conv2d_transpose(deconv2, 
                                         wd3, 
                                         output_shape=[img_batch_size, 24, 24, 3], 
                                         strides=[1, 2, 2, 1], 
                                         padding='SAME')
        deconv3 = tf.nn.sigmoid(tf.add(deconv3, bd3))
        print(deconv3.shape)
        
        decode = deconv3
    
        # calculate the loss
        loss = tf.reduce_mean(tf.square(decode - images))
    
    return decode, encoded_output, loss

def create_network(training):
    # Wrap the neural network in the scope named 'network'
    
    with tf.variable_scope('network', reuse=not training):
        # rename the input placeholder
        images = x
        
        # create tf graph for pre-processing
        images = pre_process(images=images, training=training)
        
        # create tensorflow graph for the main processing
        decode, encode, loss = main_network(images, training)
    
    return decode, encode, loss

# Create a tf variable that keeps track of the number of optimization iterations performed so far

# trainable=False means tf will not optimize this variable
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

# create the nn used for training
_, _, loss = create_network(training=True)

# create an optimizer to minimize the loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)

# create the nn used for testing
decode, _, _ = create_network(training=False)

# create the nn used for getting the encoded output
_, encode, _ = create_network(training=False)

# saver object is used for storing and retrieving variables
saver = tf.train.Saver()

train_batch_size = 64

def random_batch():
    """
    Function for creating a random batch of images from the training set
    """
    num_images = len(images_train)
    
    # creates a random index
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)
    # use the random index to select the images and their labels
    x_batch = images_train[idx, :, :, :]
    
    return x_batch

with tf.Session() as sess:
    
    num_iterations = 150000
    save_dir = 'checkpoint/cifar_10_conv_v3'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # this is the base filename for the checkpoints. TF will append iteration no etc
    save_path = os.path.join(save_dir, 'cifar_10_nn')
    
    try:
        print('Trying to print the last checkpoint...')
        
        # use tensorflow to find the latest checkpoint if any
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        
        # try and load the data in the checkpoint
        saver.restore(sess, save_path=last_chk_path)
        
        # If we get to this point, the checkpoint was successfully loaded.
        print("Restored checkpoint from:", last_chk_path)
    except:
        # If the above failed for some reason, simply
        # initialize all the variables for the TensorFlow graph.
        print("Failed to restore checkpoint. Initializing variables instead.")
        sess.run(tf.global_variables_initializer())
    
    for i in range(num_iterations):
        x_batch = random_batch()
            
        # we would also want to retrieve the global step counter
        i_global, l, _ = sess.run([global_step, loss, optimizer], feed_dict={x: x_batch})
            
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # calculate batch accuracy
            c = sess.run(loss, feed_dict={x: x_batch})
            msg = "Global Step: {0:>6}, Batch Loss: {1:>6}"
            print(msg.format(i_global, c))
            #batch_acc = sess.run(accuracy, feed_dict={x: x_batch, y_true: y_batch})

            # Print status.
            #msg = "Global Step: {0:>6}, Loss: {1:>6}, Training Batch Accuracy: {2:>6.1%}"
            #print(msg.format(i_global, l, batch_acc))
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            # Save all variables of the TensorFlow graph to a
            # checkpoint. Append the global_step counter
            # to the filename so we save the last several checkpoints.
            saver.save(sess,
                        save_path=save_path,
                        global_step=global_step)
            print("Saved checkpoint.")
    print('Optimization finished!')

test_batch_size = 5

def random_test_batch():
    """
    Function for creating a random batch of images from the test set
    """
    num_images = len(images_test)
    
    # creates a random index
    idx = np.random.choice(num_images,
                           size=test_batch_size,
                           replace=True)
    # use the random index to select the images and their labels
    x_batch = images_test[idx, :, :, :]
    y_test_cls = cls_test[idx,]
    
    return x_batch, y_test_cls

x_test_batch, y_test_cls = random_test_batch()
with tf.Session() as sess:
    saver.restore(sess, "checkpoint/cifar_10_conv_v3/cifar_10_nn-150000")
    print("Model restored.")
    
    encode_decode = sess.run(decode, feed_dict={x: x_test_batch})
    
# create figure with sub-plots
fig, ax = plt.subplots(2, 5, figsize=(8,5))

# adjusting spacing between images
fig.subplots_adjust(hspace=0.3, wspace=0.5)

for i in range(test_batch_size):
    
    # plot the input image
    ax[0][i].imshow(x_test_batch[i, :, :, :], interpolation='spline16')
    # Name of the true class
    cls_true_name = class_names[y_test_cls[i]]
    # Show the true class
    xlabel = "True: {0}".format(cls_true_name)
    # show the classes as the label on the x-axis
    ax[0][i].set_xlabel(xlabel)
    # remove the ticks from the plot
    ax[0][i].set_xticks([])
    ax[0][i].set_yticks([])
    
    # plot the reconstructed image
    ax[1][i].imshow(encode_decode[i, :, :, :])
    # remove the ticks from the plot
    ax[1][i].set_xticks([])
    ax[1][i].set_yticks([])
    
plt.draw()
plt.show()

latent_train_images = np.zeros(shape=(len(images_train), 256))
with tf.Session() as sess:
    
    saver.restore(sess, "checkpoint/cifar_10_conv_v3/cifar_10_nn-150000")
    print("Model restored.")
    
    num_images = len(images_train)
    # begin index of the next block of images
    i = 0
    batch_size = 100
    while i < num_images:
    
        # end index of the next batch
        j = min(i + batch_size, num_images)
    
        encoded_output = sess.run(encode, feed_dict={x: images_train[i:j, :]})
        
        
        latent_train_images[i:j, :] = encoded_output
        
        # set the start index of the next batch as the end index of this batch
        i = j

latent_train_images.shape

def eucledianDistance(instance1, instance2):
    return np.sqrt(np.sum(np.power((instance1 - instance2), 2)))

import operator
k = 5
test_input_index = np.random.randint(0, len(images_test) - 1)
test_feed_dict = images_test[test_input_index:(test_input_index + 1),:]

with tf.Session() as sess:
    saver.restore(sess, "checkpoint/cifar_10_conv_v3/cifar_10_nn-150000")
    print("Model restored.")
    
    encoded_test_instance = sess.run(encode, feed_dict={x: test_feed_dict})
    
distances = []
for i in range(len(images_train)):
    dist = eucledianDistance(latent_train_images[i, :], encoded_test_instance)
    distances.append((images_train[i,:], cls_train[i], dist))

distances.sort(key=operator.itemgetter(2))
neighbours = []
for j in range(k):
    neighbours.append((distances[j][0], distances[j][1]))

img = images_test[test_input_index, :]
imgplot = plt.imshow(img, interpolation='spline16')
cls_true_name = class_names[cls_test[test_input_index]]
xlabel = "True: {0}".format(cls_true_name)
plt.draw()
plt.show()
print(xlabel)

# create figure with sub-plots
fig, ax = plt.subplots(1, k, figsize=(10,5), squeeze=False)

# adjusting spacing between images
fig.subplots_adjust(hspace=0.3, wspace=0.5)

for i in range(k):
    
    # plot the input image
    ax[0][i].imshow(neighbours[i][0], interpolation='spline16')
    # Name of the true class
    cls_true_name = class_names[neighbours[i][1]]
    # Show the true class
    xlabel = "True: {0}".format(cls_true_name)
    # show the classes as the label on the x-axis
    ax[0][i].set_xlabel(xlabel)
    # remove the ticks from the plot
    ax[0][i].set_xticks([])
    ax[0][i].set_yticks([])
    
plt.draw()
plt.show()

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([[1.5],[3.4],[0.9]])
d = []
for i in range(3):
    d.append((a[i], b[i]))
d.sort(key=operator.itemgetter(1))
d

