import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm_notebook  # progress bar
import time

import docload  # convenient methods for loading and processing Project Gutenberg books

# Load and process data
files = ['../data/adventures_of_sherlock_holmes.txt',
        '../data/hound_of_the_baskervilles.txt',
        '../data/sign_of_the_four.txt']
word_array, dictionary, num_lines, num_words = docload.build_word_array(
    files, vocab_size=50000, gutenberg=True)
reverse_dict = {v: k for k, v in dictionary.items()}
print('Document loaded and processed: {} lines, {} words.'
      .format(num_lines, num_words))

# Model hyperparameters and training configuration
class Config(object):
    """Model parameters"""
    def __init__(self, num_words):
        self.vocab_size = num_words
        self.batch_size = 32
        self.num_rnn_steps = 20  # unrolled length of RNN
        self.embed_size = 64     # input embedding
        self.rnn_size = 128      # number of RNN units
        self.hidden_size = 196   # hidden layer connected to last output of RNN
        self.rui_init = 0.01     # maxval, -minval for random_uniform_initializer
        self.vsi_init = 0.01     # stddev multiplier (factor) for variance_scaling_initializer
        self.neg_samples = 64    # for noise contrastive estimation (candidate sampling loss function)
        self.learn_rate = 0.05
        self.momentum = 0.8
        self.epochs = 2
        self.embed_vis_depth = 2048  # number of word embeddings to visualize in TensorBoard

config = Config(len(dictionary))

# Aliases for especially long TensorFlow calls
rui = tf.random_uniform_initializer
vsi = tf.contrib.layers.variance_scaling_initializer
# Commonly used weight and bias initializers
rui_initializer = rui(-config.rui_init, config.rui_init, dtype=tf.float32)
vsi_initializer = vsi(factor=config.vsi_init, dtype=tf.float32)
zero_initializer = tf.zeros_initializer(dtype=tf.float32)

def feeder(config, word_array):
    """Generator. Yields training example tuples: (input, target).

    Args:
        config: Config object with model parameters.
        word_array: np.array (int), as generated by docload.build_word_array()

    Returns:
        Yields a tuple of NumPy arrays: (input, target)
    """
    batch_width = len(word_array) // config.batch_size
    # reshape data for easy slicing into shape = (batch_size, num_rnn_steps)
    data = np.reshape(word_array[0 : config.batch_size*batch_width],
                      (config.batch_size, batch_width))
    shuffle_index = [x for x in range(batch_width - config.num_rnn_steps - 1)]
    random.shuffle(shuffle_index)
    for i in shuffle_index:
        x = data[:, (i):(i+config.num_rnn_steps)]
        y = data[:, i+config.num_rnn_steps].reshape((-1, 1))
        yield (x, y)

def model(config):
    '''Embedding layer, RNN and hidden layer'''
    with tf.name_scope('embedding'):
        x = tf.placeholder(tf.int32, shape=(config.batch_size, config.num_rnn_steps), name='input')
        with tf.variable_scope('embedding', initializer=rui_initializer):
            embed_w = tf.get_variable('w', [config.vocab_size, config.embed_size])
        embed_out = tf.nn.embedding_lookup(embed_w, x, name='output')
            
    with tf.variable_scope('rnn', initializer=vsi_initializer):
        rnn_cell = tf.contrib.rnn.GRUCell(config.rnn_size, activation=tf.tanh)
        rnn_out, state = tf.nn.dynamic_rnn(rnn_cell, embed_out, dtype=tf.float32)
        
    with tf.name_scope('hidden'):
        rnn_last_output = rnn_out[:, config.num_rnn_steps-1, :]
        with tf.variable_scope('hidden'):
            hid_w = tf.get_variable('w', (config.rnn_size, config.hidden_size),
                                   initializer=vsi_initializer)
            hid_b = tf.get_variable('b', config.hidden_size, initializer=zero_initializer)
        hid_out = tf.nn.tanh(tf.matmul(rnn_last_output, hid_w) + hid_b)
            
    return hid_out, x

def loss(config, hid_out):
    """Loss Function: noise contrastive estimation on final output of RNN"""
    with tf.name_scope('output'):
        y = tf.placeholder(tf.int32, shape=(config.batch_size, 1))
        with tf.variable_scope('output'):
            w = tf.get_variable('w', (config.vocab_size, config.hidden_size),
                                   initializer=vsi_initializer)
            b = tf.get_variable('b', config.vocab_size, initializer=zero_initializer)
        batch_loss = tf.reduce_mean(
            tf.nn.nce_loss(w, b, inputs=hid_out, labels=y,
                           num_sampled=config.neg_samples,
                           num_classes=config.vocab_size,
                           num_true=1), name='batch_loss')
        tf.summary.scalar('batch_loss', batch_loss)
        # keep only top N=embed_vis_depth vectors for TensorBoard visualization:
    
    return y, batch_loss

def train(config, batch_loss):
    with tf.name_scope('optimize'):
        step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.MomentumOptimizer(config.learn_rate, config.momentum)
        train_op = optimizer.minimize(batch_loss, name='minimize_op', global_step=step)
    
    return train_op, step

class MyGraph(object):
    def __init__(self, config):
        self.hid_out, self.x = model(config)
        self.y, self.batch_loss = loss(config, self.hid_out)
        self.train_op, self.step = train(config, self.batch_loss)
        self.init = tf.global_variables_initializer()  

# Train

with tf.Graph().as_default():
    g = MyGraph(config)
    with tf.Session() as sess:
        sess.run(g.init)
        start_time = time.time()
        for e in range(config.epochs):
            for t in feeder(config, word_array):
                feed = {g.x: t[0], g.y: t[1]}
                [_, batch_loss, step] = sess.run([g.train_op, g.batch_loss, g.step],
                                               feed_dict=feed)
        print("--- %s seconds ---" % (time.time() - start_time))    

