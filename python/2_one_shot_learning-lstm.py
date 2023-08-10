import sys
import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm # just for esthetics (progression bar)
sys.path.insert(0, '../data_processing/')
from siamese_data import MNIST # load the data and process it
get_ipython().magic('matplotlib inline')

data = MNIST()

max_iter = 15000 # maximum number of iterations for training
learning_rate = 0.001
batch_train = 512 # batch size for training
batch_test = 512 # batch size for testing
display = 100 # display the training loss and accuracy every `display` step
n_test = 500 # how frequently to test the network

n_inputs = 28 # dimension of each of the input vectors
n_steps = 28 # sequence length
n_hidden = [128, 64, 64] # number of neurons of each of the LSTM cell.
n_classes = 2 # two possible classes, either `same` of `different`

with tf.device('/cpu:0'):
    x1 = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs]) # placeholder for the first network (image 1)
    x2 = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs]) # placeholder for the second network (image 2)

    # placeholder for the label. `1` for `same` and `0` for `different`.
    y = tf.placeholder(tf.int64, shape=[None])

    # placeholder for dropout (we could use different dropout for different part of the architecture)
    keep_prob = tf.placeholder(tf.float32)

def reshape_input(x_):
    """
    Reshape the inputs to match the shape requirements of the function
    `tf.nn.bidirectional_rnn`
    
    Args:
        x_: a tensor of shape `(batch_size, n_steps, n_inputs)`
        
    Returns: a `list` of length `n_steps` with its elements being tensors
    of shape `(batch_size, n_inputs)`
    """
    x_ = tf.transpose(x_, [1, 0, 2]) # shape: (n_steps, batch_size, n_inputs)
    x_ = tf.split(0, n_steps, x_) # a list of `n_steps` tensors of shape (1, batch_size, n_steps)
    return [tf.squeeze(z, [0]) for z in x_] # remove size 1 dimension --> (batch_size, n_steps)


x1_, x2_ = reshape_input(x1), reshape_input(x2)

def net(x_):
    """
    Defines the network.
    
    Args:
        x_: a tensor of shape `(batch_size, n_steps, n_inputs)` containing a batch
            of images that will be fed to one of the two networks.
    
    Returns the last states from the forward and backward cell.
    """    
    lstm_cells_fw = []
    lstm_cells_bw = []
    for hid_units in n_hidden:
        lstm_cells_fw.append(tf.nn.rnn_cell.BasicLSTMCell(hid_units, forget_bias=1.0, state_is_tuple=True))
        lstm_cells_bw.append(tf.nn.rnn_cell.BasicLSTMCell(hid_units, forget_bias=1.0, state_is_tuple=True))
    stacked_lstm_fw = tf.nn.rnn_cell.MultiRNNCell(lstm_cells_fw, state_is_tuple=True)
    stacked_lstm_bw = tf.nn.rnn_cell.MultiRNNCell(lstm_cells_bw, state_is_tuple=True)
    
    stacked_lstm_fw = tf.nn.rnn_cell.DropoutWrapper(stacked_lstm_fw, output_keep_prob=keep_prob)
    stacked_lstm_bw = tf.nn.rnn_cell.DropoutWrapper(stacked_lstm_bw, output_keep_prob=keep_prob)
    
    
    _, last_state_fw, last_state_bw = tf.nn.bidirectional_rnn(
                                        stacked_lstm_fw, stacked_lstm_bw, x_,
                                        dtype=tf.float32)
    return last_state_fw, last_state_bw

with tf.device('/gpu:0'):
    with tf.variable_scope('siamese_network') as scope:
        with tf.name_scope('network_1'):
            last_state_fw1, last_state_bw1 = net(x1_)
        with tf.name_scope('network_2'):
            scope.reuse_variables() # tied weights (reuse the weights from `network_1` for `network_2`)
            last_state_fw2, last_state_bw2 = net(x2_)

    last_state1 = []
    last_state2 = []
    for i in range(len(n_hidden)):
        for j in range(2):
            last_state1.extend([last_state_bw1[i][j], last_state_fw1[i][j]])
            last_state2.extend([last_state_bw2[i][j], last_state_fw2[i][j]])

    last_state1 = tf.concat(1, last_state1) # We concatenate the states of the first network
    last_state2 = tf.concat(1, last_state2) # We concatenate the states of the second network

    # Weights and biases for the layer that connects the outputs from the two networks
    weights = tf.get_variable('weigths_out', shape=[4 * np.sum(n_hidden), n_classes],
                    initializer=tf.random_normal_initializer(stddev=1.0/float(np.sum(n_hidden))))
    biases = tf.get_variable('biases_out', shape=[n_classes])

    # difference between the states from the two networks
    last_states_diff = tf.abs(last_state1 - last_state2) 
    logits = tf.matmul(last_states_diff, weights) + biases

with tf.device('/gpu:0'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_pred = tf.equal(tf.argmax(logits, 1), y) 
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

n_snapshot = 1000 # save the weights every `n_snapshot` step
checkpoint_dir = '../models/one_shot_learning/'
saver = tf.train.Saver() # to save the trained model and, later, to restore it.

init = tf.initialize_all_variables()

# the argument `allow_soft_placement=True` indicates that if a given function is
# not implemented for GPUs, tensorflow will automatically use its CPU counterpart.
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(init) # initialize all variables
    print('Network training begins.')
    for i in range(1, max_iter + 1):
        start = time.time()
        # We retrieve a batch of data from the training set
        batch_x1, batch_x2, batch_y = data.get_next_batch(batch_train, phase='train', one_shot=True)
        # We feed the data to the network for training
        feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: .75}
        _, loss_, accuracy_ = sess.run([optimizer, loss, accuracy], feed_dict=feed_dict)
        
        elapsed = time.time() - start
        epoch = i * batch_train / float(data.data_n)
        if i % display == 0:
            print('epoch %.2f, step %i, training loss: %.5f, training accuracy: %.3f, %.3f datum/sec' % (
                    epoch, i, loss_, accuracy_, batch_train / elapsed))
        
        # Testing the network
        if i % n_test == 0:
            # Retrieving data from the test set
            batch_x1, batch_x2, batch_y = data.get_next_batch(batch_test, phase='test', one_shot=True)
            feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: 1.0}
            accuracy_test = sess.run(accuracy, feed_dict=feed_dict)
            print('testing step %i, accuracy %.3f' % (i, accuracy_test))
            
            
        # We save a snapshot of the weights
        if i % n_snapshot == 0:
            save_path = saver.save(sess, os.path.join(checkpoint_dir,'snapshot_') + str(i) + '.ckpt')
            print('Snapshot saved in file: %s' % save_path)
            
    print('********************************')
    print('Training finished.')
    
    # testing the trained network on a large sample
    batch_x1, batch_x2, batch_y = data.get_next_batch(10000, phase='test', one_shot=True)
    feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob:1.0}
    accuracy_test = sess.run(accuracy, feed_dict=feed_dict)
    print('********************************')
    print('Testing the network.')
    print('Network accuracy %.3f' % (accuracy_test))
    print('********************************')

one_example_per_class = []
for digit in data.digits:
    one_example_per_class.append(
        getattr(data, digit + '_train')[
            np.random.randint(len(getattr(data, digit + '_train')))])

fig = plt.figure(figsize=(7,2))
for i in range(1, 11):
    a = fig.add_subplot(2, 5, i)
    a.axis('off')
    image = one_example_per_class[i - 1].reshape((28, 28)) # reshape the image from (784) to (28, 28).
    a.imshow(image, cmap='Greys_r');

def reshape_input(image):
    """
    Reshapes an image from `(784)` to `(1, 28, 28)`.
    
    Args:
        image: a `numpy array` of shape `(784)`.
    
    Returns  a `numpy array` of shape `(1, 28, 28)`.
    """
    image = np.expand_dims(image.reshape((28,28)), axis=0)
    return image

def create_benchmark(images):
    """
    Concatenates the 10 images of the benchmark into one tensor.
    
    Args:
        images: a `list` of ten `numpy array`s of shape (784).
    
    Returns a `numpy array` of shape `(10, 28, 28)`.   
    """
    images = [reshape_input(x) for x in images]
    return np.concatenate(images)
        

def duplicate_input(image):
    """
    Duplicates the image ten times.
    
    Args:
        image: a `numpy array` of shape (784).
    
    Returns a `numpy array` of shape (10, 28, 28).
    """
    image = reshape_input(image)
    image = [image for x in range(10)]
    return np.concatenate(image)

def prediction_bunch(predictions, bunch=32):
    """
    Args:
        predictions: a `numpy array` of shape `(10 * bunch, 2)`. The second
            column contains the probability that the given inputs are similar.
        bunch: an `integer` equal the to the batch size divided by 10.
    
    Returns a list of length `bunch` containing the predicted labels, i.e.
    a list of integers between 0 and 9.
    """
    predictions_ = []
    for i in range(bunch):
        predictions_.append(np.argmax(predictions[10 * i : 10 * (i + 1), 1]))
    return predictions_

def test_number(data_, benchmark, sess, bunch=32):
    """
    Args:
        data_: a `list` of `numpy array`s containing images from a specific class
            (e.g. only 5 or only 9). The images have a shape `(784)`.
        benchmark: a `numpy array` of shape `(10, 28, 28)`.
        sess: a tensorflow session.
        bunch: an `integer` equal the to the batch size divided by 10. It represents
            the number of different images being fed to the network at the same time.
    
    Returns a list of length `bunch` containing the label predictions.
    """
    benchmark_ = np.concatenate([benchmark for _ in range(bunch)])
    y_pred = []
    for i in range(0, len(data_) - bunch, bunch):
        digit1 = np.concatenate([duplicate_input(data_[j]) for j in range(i, i + bunch)])
    
        prediction_prob = tf.nn.softmax(logits)
        feed_dict = {x1: digit1, x2: benchmark_, keep_prob: 1.0}
        prediction_prob = sess.run(prediction_prob, feed_dict=feed_dict)
        y_pred.extend(prediction_bunch(prediction_prob, bunch))
    return y_pred

digit_mapping  = {i: j for (i, j) in zip(data.digits, range(10))}
benchmark = create_benchmark(one_example_per_class)

bunch = 128 # number of different images to test at the same time (batch size = 10 * bunch)

with tf.Session() as sess:
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    saver.restore(sess, latest_checkpoint)
    print('%s was restored.' % latest_checkpoint)
    for i, j in digit_mapping.iteritems():
        print i, j
        y_pred = test_number(getattr(data, i + '_test'), benchmark, sess, bunch=bunch)
        y_true = [j] * len(y_pred)
        print 'Accuracy for %i is %.3f' % (j, accuracy_score(y_true, y_pred))

def print_results(digit1_, pred):
    fig = plt.figure(figsize=(7,2))
    b = fig.add_subplot(2, 1, 1)
    b.axis('off')
    b.imshow(digit1_[0], cmap='Greys_r')
    b = fig.add_subplot(2, 1, 2)
    b.axis('off')
    b.imshow(benchmark[pred], cmap='Greys_r');

print_results(duplicate_input(data.sevens_test[221]), 7)

