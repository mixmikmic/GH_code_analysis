import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '../data_processing/')
from siamese_data import MNIST # load the data and process it
get_ipython().magic('matplotlib inline')

data = MNIST() # load the data
examples_n = 100 # display some images
indexes = np.random.choice(range(len(data.y)), examples_n, replace=False)
fig = plt.figure(figsize=(5,5))
for i in range(1, examples_n + 1):
    a = fig.add_subplot(np.sqrt(examples_n), np.sqrt(examples_n), i)
    a.axis('off')
    image = data.x[indexes[i-1]].reshape((28, 28)) # reshape the image from (784) to (28, 28).
    a.imshow(image, cmap='Greys_r');

max_iter = 1000 # maximum number of iterations for training
learning_rate = 0.001
batch_train = 128 # batch size for training
batch_test = 512 # batch size for testing
display = 50 # display the training loss and accuracy every `display` step
n_test = 200 # test the network every `n_test` step

n_inputs = 28 # dimension of each of the input vectors
n_steps = 28 # sequence length
n_hidden = 128 # number of neurons of the bi-directional LSTM
n_classes = 2 # two possible classes, either `same` of `different`

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
        
    Returns:
        A `list` of length `n_steps` with its elements being tensors of shape `(batch_size, n_inputs)`
    """
    x_ = tf.transpose(x_, [1, 0, 2]) # shape: (n_steps, batch_size, n_inputs)
    x_ = tf.split(0, n_steps, x_) # a list of `n_steps` tensors of shape (1, batch_size, n_steps)
    return [tf.squeeze(z, [0]) for z in x_] # remove size 1 dimension --> (batch_size, n_steps)


x1_, x2_ = reshape_input(x1), reshape_input(x2)

lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True) # Forwward cell
lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True) # Backward cell

lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)

with tf.variable_scope('siamese_network') as scope:
    with tf.name_scope('Bi_LSTM_1'):
        _, last_state_fw1, last_state_bw1 = tf.nn.bidirectional_rnn(
                                        lstm_fw_cell, lstm_bw_cell, x1_,
                                        dtype=tf.float32)
    with tf.name_scope('Bi_LSTM_2'):
        scope.reuse_variables() # tied weights (reuse the weights from `Bi_LSTM_1` for `Bi_LSTM_2`)
        _, last_state_fw2, last_state_bw2 = tf.nn.bidirectional_rnn(
                                        lstm_fw_cell, lstm_bw_cell, x2_,
                                        dtype=tf.float32)

# Weights and biases for the layer that connects the outputs from the two networks
weights = tf.get_variable('weigths_out', shape=[4 * n_hidden, n_classes],
                initializer=tf.random_normal_initializer(stddev=1.0/float(n_hidden)))
biases = tf.get_variable('biases_out', shape=[n_classes])

# We concatenate the states of the first network
last_state1 = tf.concat(1, [last_state_fw1[0], last_state_bw1[0],
                              last_state_fw1[1], last_state_bw1[1]])
# We concatenate the states of the second network
last_state2 = tf.concat(1, [last_state_fw2[0], last_state_bw2[0],
                              last_state_fw2[1], last_state_bw2[1]])
last_states_diff = tf.abs(last_state1 - last_state2)
logits = tf.matmul(last_states_diff, weights) + biases

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(logits, 1), y) 
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init) # initialize all variables
    print('Network training begins.')
    for i in range(1, max_iter + 1):
        # We retrieve a batch of data from the training set
        batch_x1, batch_x2, batch_y = data.get_next_batch(batch_train, phase='train')
        # We feed the data to the network for training
        feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: .9}
        _, loss_, accuracy_ = sess.run([optimizer, loss, accuracy], feed_dict=feed_dict)
        
        if i % display == 0:
            print('step %i, training loss: %.5f, training accuracy: %.3f' % (i, loss_, accuracy_))
        
        # Testing the network
        if i % n_test == 0:
            # Retrieving data from the test set
            batch_x1, batch_x2, batch_y = data.get_next_batch(batch_test, phase='test')
            feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: 1.0}
            accuracy_test = sess.run(accuracy, feed_dict=feed_dict)
            print('testing step %i, accuracy %.3f' % (i, accuracy_test))
    print('********************************')
    print('Training finished.')
    
    # testing the trained network on a large sample
    batch_x1, batch_x2, batch_y = data.get_next_batch(10000, phase='test')
    feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob:1.0}
    accuracy_test = sess.run(accuracy, feed_dict=feed_dict)
    print('********************************')
    print('Testing the network.')
    print('Network accuracy %.3f' % (accuracy_test))
    print('********************************')

