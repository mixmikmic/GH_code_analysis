# specify the path to your local clone of https://github.com/tensorflow/models,
# which is used to load ImageNet and apply VGG preprocessing
TENSORFLOW_MODELS = 'PLEASE SPECIFY'

# specify the path of the ImageNet tfrecords files
IMAGENET_DATA = '/gpfs01/bethge/data/imagenet'

# specify the path to your initial checkpoint to restore pretrained models, e.g. VGG19
INITIAL_CHECKPOINT = '/gpfs01/bethge/data/tf-model-checkpoints/vgg_19.ckpt'

# specify the directory where checkpoints and summaries are stored; start TensorBoard with access to this directory
LOGDIR = 'PLEASE SPECIFY'

# ipython configuration
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

# standard library
import sys
import os

# third-party packages
import tensorflow as tf
slim = tf.contrib.slim

# local modules
IMPORT_PATHS = [os.path.join(TENSORFLOW_MODELS, 'slim')]
sys.path.extend(set(IMPORT_PATHS) - set(sys.path))
from datasets import imagenet
from preprocessing import vgg_preprocessing

# this is NOT the same as the vgg_19 in tensorflow/models or tf.contrib.slim
def vgg_19(inputs,
           is_training=True,
           dropout_keep_prob=0.5,
           scope='vgg_19',
           reuse=False):
    """VGG19 implementation using fully-connected layers
    
    Fully-connected layers are currently faster than 1x1 convolutions
    and should be used when VGG is part of a training pipeline. During
    evaluation, you might want to use the corresponding fully-convolution
    network to be able to apply it to other image sizes.
    """
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], data_format='NCHW'):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
            net = slim.fully_connected(net, 1000, activation_fn=None, normalizer_fn=None, scope='fc8')
            return net

def get_training_data():
    with tf.device('/cpu:0'):
        dataset = imagenet.get_split('train', IMAGENET_DATA)
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=2, # controls the speed at which data is loaded (can be much lower than num_threads)
            shuffle=True,
            common_queue_capacity=512,
            common_queue_min=128)
        image, label = provider.get(['image', 'label'])
        
        # preprocess the image
        image = vgg_preprocessing.preprocess_for_train(
            image,
            224,
            224,
            resize_side_min=256,
            resize_side_max=512)
        
        # NHWC to NCHW
        image = tf.transpose(image, [2, 0, 1])
        
        # preprocess the label
        label = tf.subtract(label, 1) # 1..1000 to 0..999

    images, labels = tf.train.batch(
        [image, label],
        batch_size=64, # specify the batch size here
        num_threads=16, # controls the speed at which images are preprocessed
        capacity=128)
    return images, labels

def top_k_accuracy(labels, predictions, k, name='top_k_accuracy'):
    """Something like this should be in tf.metrics, but as far as I can see, there is no such function."""
    with tf.name_scope(name, 'top_k_accuracy'):
        correct = tf.nn.in_top_k(predictions, labels, k=k)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy

g = tf.Graph()
with g.as_default():
    # load the data
    images, labels = get_training_data()
    
    # apply the model
    predictions = vgg_19(images, is_training=True)
    
    # define the loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels, predictions)
    total_loss = tf.losses.get_total_loss()
    
    # define the metrics
    top5_accuracy = top_k_accuracy(labels, predictions, k=5)
    top1_accuracy = top_k_accuracy(labels, predictions, k=1)
    
    # define the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
    
    # create the train op
    train_op = slim.learning.create_train_op(total_loss, optimizer) # specify variables to train here
    
    # create summaries
    tf.summary.histogram('predictions', predictions)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('top5_accuracy', top5_accuracy)
    tf.summary.scalar('top1_accuracy', top1_accuracy)
    
    # define an init function that restores the pretrained VGG
    init_fn = slim.assign_from_checkpoint_fn(
        INITIAL_CHECKPOINT,
        slim.get_model_variables(),
        reshape_variables=True) # reshape variables because the checkpoint is for a fully-convolutional network

# this will run forever: stop it using Kernel -> Interrupt
# the first few steps will take longer, until the queues are filled
slim.learning.train(
    train_op,
    LOGDIR,
    graph=g,
    init_fn=init_fn,
    log_every_n_steps=1, # increase to avoid too many log statements
    save_summaries_secs=20)

