import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '../data_processing/')
from siamese_data import MNIST # load the data and process it
get_ipython().magic('matplotlib inline')

data = MNIST()

n_classes = 2 # two possible classes, either `same` of `different`

x1 = tf.placeholder(tf.float32, shape=[None, 28, 28, 1]) # placeholder for the first network (image 1)
x2 = tf.placeholder(tf.float32, shape=[None, 28, 28, 1]) # placeholder for the second network (image 2)

# placeholder for the label. `[1, 0]` for `same` and `[0, 1]` for `different`.
y = tf.placeholder(tf.int64, shape=[None])

# placeholder for dropout (we could use different dropouts for different part of the architecture)
keep_prob = tf.placeholder(tf.float32)

def conv_2d(x, kernel_shape, stride, padding='SAME'):
    strides = [1, stride, stride, 1]
    K = tf.get_variable(name='weights', shape=kernel_shape, initializer=tf.random_normal_initializer())
    b = tf.get_variable(name='biases', shape=kernel_shape[-1], initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv2d(x, K, strides=strides, padding=padding)
    conv += b
    print(conv.get_shape())
    return tf.nn.relu6(conv, name='relu6')

def maxpool(x, pool, stride, padding='SAME'):
    ksize = [1, pool, pool, 1]
    strides = [1, stride, stride, 1]
    maxpool = tf.nn.max_pool(x, ksize, strides, padding=padding, name='maxpool')
    print(maxpool.get_shape())
    return maxpool


def convnet(x):
    with tf.variable_scope('conv1') as scp: # (None, 28, 28, 1)
        net = conv_2d(x, [5, 5, 1, 32], stride=1) # (None, 28, 28, 32)
        net = maxpool(net, 2, 2, padding='VALID') # (None, 14, 14, 32)
    
    with tf.variable_scope('conv2') as scp:
        net = conv_2d(net, [3, 3, 32, 32], stride=1) # (None, 14, 14, 32)
        net = maxpool(net, 2, 2) # (None, 7, 7, 32)
        
    #with tf.variable_scope('conv3') as scp:
    #    net = conv_2d(net, [5, 5, 64, 64], stride=1) # (None, 6, 6, 64)
    #    net = maxpool(net, 2, 2) # (None, 3, 3, 64)
        
    with tf.variable_scope('fully_connected') as scp:
        W = tf.get_variable('weights', shape=[7*7*32, 512], initializer=tf.random_normal_initializer())
        b = tf.get_variable('biases', shape=[512], initializer=tf.constant_initializer(0.))
        net = tf.reshape(net, shape=[-1, 7*7*32])
        net = tf.matmul(net, W) + b # (None, 512)
        print(net.get_shape())    
    return net

with tf.variable_scope('siamese_network') as scope:
    with tf.name_scope('convnet_1'):
        convnet1 = convnet(x1)
    with tf.name_scope('convnet_2'):
        scope.reuse_variables() # tied weights (reuse the weights from `Bi_LSTM_1` for `Bi_LSTM_2`)
        convnet2 = convnet(x2)

with tf.name_scope('output') as scp:
    net_difference = tf.abs(convnet1 - convnet2)
    W = tf.get_variable('weights_' + scp, shape=[512, n_classes], initializer=tf.random_normal_initializer())
    b = tf.get_variable('biases_' + scp, shape=[n_classes], initializer=tf.constant_initializer(0.))
    logits = tf.matmul(net_difference, W) + b # (None, 2)
    print(logits.get_shape())

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer().minimize(loss)

correct_pred = tf.equal(tf.argmax(logits, 1), y) 
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

max_iter = 10000 # maximum number of iterations for training
batch_train = 128 # batch size for training
batch_test = 512 # batch size for testing
display = 50 # display the training loss and accuracy every `display` step
n_test = 100 # test the network every `n_test` step

n_snapshot = 1000 # save the weights every `n_snapshot` step
checkpoint_dir = 'models/one_shot_learning/'
saver = tf.train.Saver() # to save the trained model

with tf.Session() as sess:
    sess.run(init) # initialize all variables
    print('Network training begins.')
    for i in range(1, max_iter + 1):
        # We retrieve a batch of data from the training set (digits between 0 and 6)
        batch_x1, batch_x2, batch_y = data.get_next_batch(batch_train, phase='train', one_shot=True)
        batch_x1, batch_x2 = np.expand_dims(batch_x1, axis=3), np.expand_dims(batch_x2, axis=3)
        # We feed the data to the network for training
        feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: .9}
        _, loss_, accuracy_ = sess.run([optimizer, loss, accuracy], feed_dict=feed_dict)
        
        if i % display == 0:
            print('step %i, training loss: %.5f, training accuracy: %.3f' % (i, loss_, accuracy_))
        
        # Testing the network
        if i % n_test == 0:
            # Retrieving data from the test set
            batch_x1, batch_x2, batch_y = data.get_next_batch(batch_test, phase='test', one_shot=True)
            batch_x1, batch_x2 = np.expand_dims(batch_x1, axis=3), np.expand_dims(batch_x2, axis=3)
            feed_dict = {x1: batch_x1, x2: batch_x2, y: batch_y, keep_prob: 1.0}
            accuracy_test = sess.run(accuracy, feed_dict=feed_dict)
            print('testing step %i, accuracy %.3f' % (i, accuracy_test))
        
        # We save a snapshot of the weights
        if i % n_snapshot == 0:
            save_path = saver.save(sess, os.path.join(checkpoint_dir,'snapshot_') + str(i) + '.ckpt')
            print('Snapshot saved in file: %s' % save_path)

    print('********************************')
    print('Training finished.')

print batch_x1.shape
print np.expand_dims(batch_x1, axis=3).shape

one_example_per_class = []
for digit in data.digits:
    one_example_per_class.append(getattr(data, digit)[np.random.randint(len(getattr(data, digit)))])

fig = plt.figure(figsize=(7,2))
for i in range(1, 11):
    a = fig.add_subplot(2, 5, i)
    a.axis('off')
    image = one_example_per_class[i - 1].reshape((28, 28)) # reshape the image from (784) to (28, 28).
    a.imshow(image, cmap='Greys_r');

def reshape_input(image):
    """
    Takes a `numpy array` of shape (784) and reshape it
    into a `numpy array` of shape (1, 28, 28).
    """
    image = np.expand_dims(image.reshape((28,28)), axis=0)
    return np.expand_dims(image, axis=3)

def reshape_label(label):
    """
    Argss:
        label: a list of two elements.
    """
    if label[0] == label[1]:
        return np.expand_dims(np.asarray(1),axis=0)
    else:
        return np.expand_dims(np.asarray(0),axis=0)

def compare_two_digits(digit1, digit2, label1, label2, sess):
    true_y = 'different'
    if label1 == label2:
        true_y = 'same'
    label = reshape_label([label1, label2])
    feed_dict = {x1: digit1, x2: digit2, y: label, keep_prob: 1.0}
    logits_, accuracy_test = sess.run([logits, accuracy], feed_dict=feed_dict)
    predicted_y = 'different'
    if np.argmax(logits_[0]) == 1:
         predicted_y = 'same'
    return true_y, predicted_y

one_example_per_class = [reshape_input(x) for x in one_example_per_class]

checkpoint_dir = 'models/one_shot_learning/'
with tf.Session() as sess:
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    saver.restore(sess, latest_checkpoint)
    print('%s was restored.' % latest_checkpoint)
    digit1 = reshape_input(data.eights_test[222])
    for i in range(10):
        digit2 = reshape_input(one_example_per_class[i])
        true_y, predicted_y = compare_two_digits(digit1, digit2, 8, i, sess)
        print 'Comparing %i with 7:' %i, predicted_y

a = [[ 2.24756432,  0.03931087]]
a[0]

