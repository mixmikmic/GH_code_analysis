get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
#import TensorFlow and MNIST
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load data and targets as one-hot vectors
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS

N_CLASSES = 10

MAX_ITERS = 150

def show_images(train_images, test_images, n_images = 5):
    (fig, ax) = plt.subplots(nrows = 2, ncols = n_images)
    for i in range(n_images):
        train_image = train_images[i].reshape(
            (IMAGE_HEIGHT, IMAGE_WIDTH))
        ax[0, i].imshow(train_image, cmap = 'Greys')
        ax[0, i].set_axis_off()
        
        test_image = test_images[i].reshape(
            (IMAGE_HEIGHT, IMAGE_WIDTH))
        ax[1, i].imshow(test_image, cmap = 'Greys')
        ax[1, i].set_axis_off()

plt.show()

print('Image shape {}'.format(mnist.train.images[0].shape))
print('Total of images for training {}'.format(len(mnist.train.images)))
print('Total of images for testing {}'.format(len(mnist.test.images)))
show_images(mnist.train.images, mnist.test.images)

def evaluate(model_output, feed_dict = {}):
    # number of samples for evaluation
    n_samples = 500
    
    feed_dict[x] = mnist.test.images[:n_samples]
    feed_dict[y] = mnist.test.labels[:n_samples]
    
    correct_predictions = tf.equal(tf.argmax(model_output, 1), 
                                  tf.argmax(y, 1))
    accuracy_model = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32))
    accuracy = accuracy_model.eval(feed_dict)
    print('Model accuracy: {}'.format(accuracy)) 

BATCH_SIZE = 64

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE])
    # a placeholder for labels, shape = (N_samples, n_classes)
    y = tf.placeholder(tf.float32, shape = [None, N_CLASSES])
    
    # weight matrix with shape (n_inputs, n_outputs)
    # initialized using a normal distribution
    W = tf.Variable(
        tf.truncated_normal([IMAGE_SIZE, N_CLASSES], stddev = 0.1))
    # bias vector with shape (n_outputs)
    b = tf.Variable(tf.zeros([N_CLASSES]))
    
    # linear model
    logits = tf.matmul(x, W) + b
    loss = loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels = y, logits = logits))
    
    # optimization with a learning rate decay every 100 i
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 
        decay_steps = 100, decay_rate = 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.minimize(loss, global_step = global_step)
    
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    for iteration in range(MAX_ITERS):
        (images, labels) = mnist.train.next_batch(BATCH_SIZE)
        (_, iter_loss) = session.run([optimizer, loss], 
                              feed_dict = {x: images,
                                          y: labels})
    
        if iteration % 10 == 0:
            print('Iteration {}, loss = {}.'.format(
                iteration, iter_loss))
            
    model = tf.nn.softmax(logits)
    evaluate(model)

# create a weight matrix
def weight_matrix(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# create a bias vector
def bias_vector(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# create a convolution kernel
def conv2d(x, W, h_stride = 1, w_stride = 1, padding = 'SAME'):
    return tf.nn.conv2d(x, W, 
                        # N, H, W, C
                        strides = [1, h_stride, w_stride, 1],
                        padding = padding)

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE])
    shape = [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    x_image = tf.reshape(x, shape)
    # a placeholder for labels, shape = (N_samples, n_classes)
    y = tf.placeholder(tf.float32, shape = [None, N_CLASSES])
    
    conv1_channels = 32
    # convolution W matrix [k_h, k_w, input_n_c, n_c]
    W_conv1 = weight_matrix([5, 5, IMAGE_CHANNELS, conv1_channels])
    b_conv1 = bias_vector([conv1_channels])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
    conv2_channels = 64
    # convolution W matrix [k_h, k_w, input_n_c, n_c]
    W_conv2 = weight_matrix([5, 5, conv1_channels, conv2_channels])
    b_conv2 = bias_vector([conv2_channels])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    conv2_output_size = IMAGE_HEIGHT * IMAGE_WIDTH * conv2_channels
    shape = [-1, conv2_output_size]
    h_conv2_flat = tf.reshape(h_conv2, shape)
    
    W_fc1 = weight_matrix([conv2_output_size, 1024])
    b_fc1 = bias_vector([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    
    W_fc2 = weight_matrix([1024, N_CLASSES])
    b_fc2 = bias_vector([N_CLASSES])
    
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = y, 
                                                logits = logits))
    
    # optimization with a learning rate decay every 100 i
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 
        decay_steps = 100, decay_rate = 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.minimize(loss, global_step = global_step)
    
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    for iteration in range(MAX_ITERS):
        (images, labels) = mnist.train.next_batch(BATCH_SIZE)
        (_, iter_loss) = session.run([optimizer, loss], 
                              feed_dict = {x: images,
                                          y: labels})
    
        if iteration % 10 == 0:
            print('Iteration {}, loss = {}.'.format(
                iteration, iter_loss))
            
    model = tf.nn.softmax(logits)
    evaluate(model)

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE])
    shape = [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    x_image = tf.reshape(x, shape)
    # a placeholder for labels, shape = (N_samples, n_classes)
    y = tf.placeholder(tf.float32, shape = [None, N_CLASSES])
    
    conv1_channels = 32
    # convolution W matrix [k_h, k_w, input_n_c, n_c]
    W_conv1 = weight_matrix([5, 5, IMAGE_CHANNELS, conv1_channels])
    b_conv1 = bias_vector([conv1_channels])
    h_conv1 = conv2d(x_image, W_conv1, padding = 'VALID') + b_conv1
    h_conv1 = tf.nn.relu(h_conv1)
    
    conv2_channels = 64
    # convolution W matrix [k_h, k_w, input_n_c, n_c]
    W_conv2 = weight_matrix([5, 5, conv1_channels, conv2_channels])
    b_conv2 = bias_vector([conv2_channels])
    h_conv2 = conv2d(h_conv1, W_conv2, padding = 'VALID') + b_conv2
    h_conv2 = tf.nn.relu(h_conv2)
    conv2_output_size = 20 * 20 * conv2_channels
    shape = [-1, conv2_output_size]
    h_conv2_flat = tf.reshape(h_conv2, shape)
    
    W_fc1 = weight_matrix([conv2_output_size, 1024])
    b_fc1 = bias_vector([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    
    W_fc2 = weight_matrix([1024, N_CLASSES])
    b_fc2 = bias_vector([N_CLASSES])
    
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = y, 
                                                logits = logits))
    
    # optimization with a learning rate decay every 100 i
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 
        decay_steps = 100, decay_rate = 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.minimize(loss, global_step = global_step)
    
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    for iteration in range(MAX_ITERS):
        (images, labels) = mnist.train.next_batch(BATCH_SIZE)
        (_, iter_loss) = session.run([optimizer, loss], 
                              feed_dict = {x: images,
                                          y: labels})
    
        if iteration % 10 == 0:
            print('Iteration {}, loss = {}.'.format(
                iteration, iter_loss))
            
    model = tf.nn.softmax(logits)
    evaluate(model)

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE])
    shape = [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    x_image = tf.reshape(x, shape)
    # a placeholder for labels, shape = (N_samples, n_classes)
    y = tf.placeholder(tf.float32, shape = [None, N_CLASSES])
    
    conv1_channels = 32
    # convolution W matrix [k_h, k_w, input_n_c, n_c]
    W_conv1 = weight_matrix([5, 5, IMAGE_CHANNELS, conv1_channels])
    b_conv1 = bias_vector([conv1_channels])
    h_conv1 = conv2d(x_image, W_conv1, 
                     h_stride = 2, w_stride = 2) + b_conv1
    h_conv1 = tf.nn.relu(h_conv1)
    
    conv2_channels = 64
    # convolution W matrix [k_h, k_w, input_n_c, n_c]
    W_conv2 = weight_matrix([5, 5, conv1_channels, conv2_channels])
    b_conv2 = bias_vector([conv2_channels])
    h_conv2 = conv2d(h_conv1, W_conv2, 
                     h_stride = 2, w_stride = 2) + b_conv2
    h_conv2 = tf.nn.relu(h_conv2)
    conv2_output_size = tf.to_int32(np.prod(h_conv2.shape[1:]))
    shape = [-1, conv2_output_size]
    h_conv2_flat = tf.reshape(h_conv2, shape)
    
    W_fc1 = weight_matrix([conv2_output_size, 1024])
    b_fc1 = bias_vector([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    
    W_fc2 = weight_matrix([1024, N_CLASSES])
    b_fc2 = bias_vector([N_CLASSES])
    
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = y, 
                                                logits = logits))
    
    # optimization with a learning rate decay every 100 i
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 
        decay_steps = 100, decay_rate = 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.minimize(loss, global_step = global_step)
    
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    for iteration in range(MAX_ITERS):
        (images, labels) = mnist.train.next_batch(BATCH_SIZE)
        (_, iter_loss) = session.run([optimizer, loss], 
                              feed_dict = {x: images,
                                          y: labels})
    
        if iteration % 10 == 0:
            print('Iteration {}, loss = {}.'.format(
                iteration, iter_loss))
            
    model = tf.nn.softmax(logits)
    evaluate(model)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], 
        strides = [1, 2, 2, 1], padding = 'SAME')

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE])
    shape = [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    x_image = tf.reshape(x, shape)
    # a placeholder for labels, shape = (N_samples, n_classes)
    y = tf.placeholder(tf.float32, shape = [None, N_CLASSES])
    
    conv1_channels = 32
    # convolution W matrix [k_h, k_w, input_n_c, n_c]
    W_conv1 = weight_matrix([5, 5, IMAGE_CHANNELS, conv1_channels])
    b_conv1 = bias_vector([conv1_channels])
    h_conv1 = conv2d(x_image, W_conv1) + b_conv1
    h_conv1 = tf.nn.relu(h_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    conv2_channels = 64
    # convolution W matrix [k_h, k_w, input_n_c, n_c]
    W_conv2 = weight_matrix([5, 5, conv1_channels, conv2_channels])
    b_conv2 = bias_vector([conv2_channels])
    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_conv2 = tf.nn.relu(h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    pool2_output_size = tf.to_int32(np.prod(h_pool2.shape[1:]))
    h_pool2_flat = tf.reshape(h_pool2, [-1, pool2_output_size])
    
    W_fc1 = weight_matrix([pool2_output_size, 1024])
    b_fc1 = bias_vector([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    W_fc2 = weight_matrix([1024, N_CLASSES])
    b_fc2 = bias_vector([N_CLASSES])
    
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = y, 
                                                logits = logits))
    
    # optimization with a learning rate decay every 100 i
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 
        decay_steps = 100, decay_rate = 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.minimize(loss, global_step = global_step)
    
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    for iteration in range(MAX_ITERS):
        (images, labels) = mnist.train.next_batch(BATCH_SIZE)
        (_, iter_loss) = session.run([optimizer, loss], 
                              feed_dict = {x: images,
                                          y: labels})
    
        if iteration % 10 == 0:
            print('Iteration {}, loss = {}.'.format(
                iteration, iter_loss))
            
    model = tf.nn.softmax(logits)
    evaluate(model)

beta = 0.0005

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE])
    shape = [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    x_image = tf.reshape(x, shape)
    # a placeholder for labels, shape = (N_samples, n_classes)
    y = tf.placeholder(tf.float32, shape = [None, N_CLASSES])
    
    conv1_channels = 32
    # convolution W matrix [k_h, k_w, input_n_c, n_c]
    W_conv1 = weight_matrix([5, 5, IMAGE_CHANNELS, conv1_channels])
    b_conv1 = bias_vector([conv1_channels])
    h_conv1 = conv2d(x_image, W_conv1) + b_conv1
    h_conv1 = tf.nn.relu(h_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    conv2_channels = 64
    # convolution W matrix [k_h, k_w, input_n_c, n_c]
    W_conv2 = weight_matrix([5, 5, conv1_channels, conv2_channels])
    b_conv2 = bias_vector([conv2_channels])
    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_conv2 = tf.nn.relu(h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    pool2_output_size = tf.to_int32(np.prod(h_pool2.shape[1:]))
    h_pool2_flat = tf.reshape(h_pool2, [-1, pool2_output_size])
    
    W_fc1 = weight_matrix([pool2_output_size, 1024])
    b_fc1 = bias_vector([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    W_fc2 = weight_matrix([1024, N_CLASSES])
    b_fc2 = bias_vector([N_CLASSES])
    
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = y, 
                                                logits = logits))
    
    regularization = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
        tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))
    
    loss = tf.reduce_mean(loss + beta * regularization)
    
    # optimization with a learning rate decay every 100 iterations
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 
        decay_steps = 100, decay_rate = 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.minimize(loss, global_step = global_step)
    
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    for iteration in range(MAX_ITERS):
        (images, labels) = mnist.train.next_batch(BATCH_SIZE)
        (_, iter_loss) = session.run([optimizer, loss], 
                              feed_dict = {x: images,
                                          y: labels})
    
        if iteration % 10 == 0:
            print('Iteration {}, loss = {}.'.format(
                iteration, iter_loss))
            
    model = tf.nn.softmax(logits)
    evaluate(model)

beta = 0.0005

graph = tf.Graph()
with graph.as_default():
    # a placeholder for inputs, shape = (N_samples, n_features)
    x = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE])
    shape = [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    x_image = tf.reshape(x, shape)
    # a placeholder for labels, shape = (N_samples, n_classes)
    y = tf.placeholder(tf.float32, shape = [None, N_CLASSES])
    
    conv1_channels = 32
    # convolution W matrix [k_h, k_w, input_n_c, n_c]
    W_conv1 = weight_matrix([5, 5, IMAGE_CHANNELS, conv1_channels])
    b_conv1 = bias_vector([conv1_channels])
    h_conv1 = conv2d(x_image, W_conv1) + b_conv1
    h_conv1 = tf.nn.relu(h_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    conv2_channels = 64
    # convolution W matrix [k_h, k_w, input_n_c, n_c]
    W_conv2 = weight_matrix([5, 5, conv1_channels, conv2_channels])
    b_conv2 = bias_vector([conv2_channels])
    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_conv2 = tf.nn.relu(h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    pool2_output_size = tf.to_int32(np.prod(h_pool2.shape[1:]))
    h_pool2_flat = tf.reshape(h_pool2, [-1, pool2_output_size])
    
    W_fc1 = weight_matrix([pool2_output_size, 1024])
    b_fc1 = bias_vector([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_matrix([1024, N_CLASSES])
    b_fc2 = bias_vector([N_CLASSES])
    
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = y, 
                                                logits = logits))
    
    regularization = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
        tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))
    
    loss = tf.reduce_mean(loss + beta * regularization)
    
    # optimization with a learning rate decay every 100 iterations
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 
        decay_steps = 100, decay_rate = 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.minimize(loss, global_step = global_step)
    
with tf.Session(graph = graph) as session:
    tf.global_variables_initializer().run()
    for iteration in range(MAX_ITERS):
        (images, labels) = mnist.train.next_batch(BATCH_SIZE)
        (_, iter_loss) = session.run([optimizer, loss], 
                              feed_dict = {x: images,
                                          y: labels,
                                          keep_prob: 0.5})
    
        if iteration % 10 == 0:
            print('Iteration {}, loss = {}.'.format(
                iteration, iter_loss))
            
    model = tf.nn.softmax(logits)
    evaluate(model, {keep_prob: 1.0})



