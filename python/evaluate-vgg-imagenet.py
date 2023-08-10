# specify the path to your local clone of https://github.com/tensorflow/models,
# which is used to load ImageNet and apply VGG preprocessing
TENSORFLOW_MODELS = 'PLEASE SPECIFY'

# specify the path of the ImageNet tfrecords files
IMAGENET_DATA = '/gpfs01/bethge/data/imagenet'

# specify the path to your initial checkpoint to restore pretrained models, e.g. VGG19
INITIAL_CHECKPOINT = '/gpfs01/bethge/data/tf-model-checkpoints/vgg_19.ckpt'

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
import tqdm

# local modules
IMPORT_PATHS = [os.path.join(TENSORFLOW_MODELS, 'slim')]
sys.path.extend(set(IMPORT_PATHS) - set(sys.path))
from datasets import imagenet
from preprocessing import vgg_preprocessing

# this is NOT the same as the vgg_19 in tensorflow/models or tf.contrib.slim
def vgg_19(inputs,
           is_training,
           dropout_keep_prob=0.5,
           scope='vgg_19',
           reuse=False):
    """VGG19 implementation using fully-connected layers
    
    This is an implementation of VGG19 using fully-connected layers rather
    than 1x1 convolutions, because fully-connected layers are slightly faster.
    To evaluate images of arbitrary size, replace this with a
    fully-convolutional network definition.
    """
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
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

def get_validation_data():
    with tf.device('/cpu:0'):
        dataset = imagenet.get_split('validation', IMAGENET_DATA)
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=1, # controls the speed at which data is loaded
            shuffle=False,
            common_queue_capacity=256,
            common_queue_min=128)
        image, label = provider.get(['image', 'label'])
        
        # preprocess the image
        image = vgg_preprocessing.preprocess_for_eval(
            image,
            224,
            224,
            resize_side=256)
        
        # preprocess the label
        label = tf.subtract(label, 1) # 1..1000 to 0..999

    images, labels = tf.train.batch(
        [image, label],
        batch_size=64, # specify the batch size here
        num_threads=16, # controls the speed at which images are preprocessed
        capacity=256)
    return images, labels

g = tf.Graph()
with g.as_default():
    # load the data
    images, labels = get_validation_data()
    
    # apply the model
    predictions = vgg_19(images, is_training=False)
    
    # define the metrics
    top5_accuracy, top5_accuracy_update = tf.metrics.recall_at_k(labels, predictions, k=5)
    
    # initialize local variables used by recall_at_k
    init_op = tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer())
    
    # restore model variables
    restorer = tf.train.Saver(slim.get_model_variables(), reshape=True)

with tf.Session(graph=g) as sess:
    sess.run(init_op)
    restorer.restore(sess, INITIAL_CHECKPOINT)
    with slim.queues.QueueRunners(sess):
        n_batches = 10
        for _ in tqdm.trange(n_batches):
            sess.run(top5_accuracy_update)
            
        print(sess.run(top5_accuracy))



