# Import Dependencies
import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Unpickle Data [Train + Test]
def load_data(file):
    with open(file,'rb') as f:
        cifar_dict = pickle.load(f, encoding='bytes')
    return cifar_dict

# Load Data
dataset = [0,1,2,3,4,5,6]

dataset[0] = load_data('./cifar-10-batches-py/batches.meta')
dataset[1] = load_data('./cifar-10-batches-py/data_batch_1')
dataset[2] = load_data('./cifar-10-batches-py/data_batch_2')
dataset[3] = load_data('./cifar-10-batches-py/data_batch_3')
dataset[4] = load_data('./cifar-10-batches-py/data_batch_4')
dataset[5] = load_data('./cifar-10-batches-py/data_batch_5')
dataset[6] = load_data('./cifar-10-batches-py/test_batch')

# Separate data into training and test
batches = dataset[0]
data_batch1 = dataset[1]
data_batch2 = dataset[2]
data_batch3 = dataset[3]
data_batch4 = dataset[4]
data_batch5 = dataset[5]
test_data = dataset[6]

# Metadata for Dataset
batches

data_batch1.keys()

# Check the Shape and Size of Images
data_batch1[b'data'].shape

# Size of a single image
data_batch1[b'data'][0].shape

# Display a single image
# Load image and reshape
X = data_batch1[b'data']
X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype('uint8')

# Shape of single image after processing
X[0].shape

# Plot single image
plt.imshow(X[0])

# Reshape IMage by Image
X = data_batch1[b'data']

all_images = X.reshape(10000,3,32,32)

sample_image = all_images[0]

sample_image

sample_image.shape

# Tranpose the image
# Take value at axis 1 and 2 in original image i.e. 32,32 to axis 0 and 1.  =>  (3,32,32)  => (32,32,)
# Take value from axis 0 to axis 2 i.e. 3 to axis 2. =>  (3,32,32)  => (32,32,3)
sample_image.transpose(1,2,0)

sample_image.transpose(1,2,0).shape

plt.imshow(sample_image.transpose(1,2,0))

# One Hot Encoding for Labels
def one_hot_encode(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

class CifarHelper():
    
    def __init__(self):
        self.i = 0
        self.train_data = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
        self.test_data = [test_data]
        
        # Initialize Variables
        self.training_images = None
        self.training_labels = None
        
        self.test_images = None
        self.test_labels = None
    
    # Normalize and Reshape Images
    def set_up_images(self):
        print('Setting up Training Images and Lables !!')
        
        # Vertically stack training images
        self.training_images = np.vstack([d[b'data'] for d in self.train_data])
        train_len = len(self.training_images)
        
        # Reshape and Normalize Training Images
        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0,2,3,1) / 255
        self.training_labels = one_hot_encode(np.hstack([d[b'labels'] for d in self.train_data]))
        
        print('Setting up Test Images and Lables !!')
        
        # Vertically stack training images
        self.test_images = np.vstack([d[b'data'] for d in self.test_data])
        test_len = len(self.test_images)
        
        # Reshape and Normalize Training Images
        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0,2,3,1) / 255
        self.test_labels = one_hot_encode(np.hstack([d[b'labels'] for d in self.test_data]), 10)
    
    # Make Data Batches of Batch Size = 100
    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i+batch_size].reshape(100,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

# Load the CIFAR Dataset using Helper Class
ch = CifarHelper()
ch.set_up_images()

# Create the Tensorflow Model

# Features
X = tf.placeholder(tf.float32, shape=[None, 32,32,3])

# Labels
y = tf.placeholder(tf.float32, shape=[None, 10])

# Hold Probability for Dropout
hold_probb = tf.placeholder(tf.float32)

# Helper Functions

# Initialize Weights
def init_weights(shape):
    # Create Random Distribution with "Mean = 0" and "Standard Deviation = 0.1"
    init_random_dist = tf.truncated_normal(shape, mean=0, stddev=0.1)
    return tf.Variable(init_random_dist)


# Initialize Bias
def init_bias(shape):
    # All bias values initialized to constant value = 0.1
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)


# CONV2D
def conv2D(x,W,stride=[1,1,1,1],pad='SAME'):
    # x: Input Tensor, shape = [batch, Height, Width, Channels]
    # W (Kernel): [filter Height, filter Width, Channels In, Channels Out]
    # 'SAME': Pad with 0's, 'VALID': 
    return tf.nn.conv2d(x, W, strides=stride, padding=pad)


# Pooling Layer
def max_pooling(x, ks=[1,2,2,1], stride=[1,2,2,1], pad='SAME'):
    # "x" : Input Tensor, shape = [batch, Height, Width, Channels]
    # "ksize" : Size of Window for each dimension of I/P Tensor 
    return tf.nn.max_pool(x,ksize=ks, strides=stride, padding=pad)


# Convolutional Layer
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2D(input_x, W) + b)


# Fully Connected Layer
def fully_connected_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer,W) + b

# Create CNN Layers
# Layer-1
conv1 = convolutional_layer(X, shape=[4,4,3,32])
pool1 = max_pooling(conv1)

# Layer-2
conv2 = convolutional_layer(pool1, shape=[4,4,32,64])
pool2 = max_pooling(conv2)

# Flatten Layer
flatten = tf.reshape(pool2, [-1,8*8*64])

# Fully Connected Layer
full_layer = tf.nn.relu(fully_connected_layer(flatten,1024))

# Dropout Layer
dropout_layer = tf.nn.dropout(full_layer, keep_prob=hold_probb)

# Output Layer
y_pred = fully_connected_layer(dropout_layer,10)

# Loss Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

# Initialize All Variables
init = tf.global_variables_initializer()

# Run Session
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(5000):
        # Get Images in Batch of 100 Images
        batch = ch.next_batch(batch_size=100)
        
        sess.run(optimizer, feed_dict={X:batch[0], y: batch[1], hold_probb: 0.5})
        
        if i%100 == 0:
            matches = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            print("STEP: {0}, ACCURACY: {1}".format(i, sess.run(acc,feed_dict={X:ch.test_images, y:ch.test_labels, hold_probb: 1.0})))
    print("\nModel Training Completed !!!")

