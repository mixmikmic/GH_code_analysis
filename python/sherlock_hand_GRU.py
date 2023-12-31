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
one_initializer = tf.ones_initializer(dtype=tf.float32)

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
        
def epoch_len(config, word_array):
    """Number of training steps in an epoch. Used for progress bar"""
    batch_width = len(word_array) // config.batch_size
    return batch_width - config.num_rnn_steps - 1

def init_rnn_cell(x, num_cells, batch_size):
    """Set up variables for re-use"""
    i_sz = x.shape[1]+num_cells
    o_sz = num_cells
    with tf.variable_scope('GRU'):
        Wr = tf.get_variable('Wr', (i_sz, o_sz), tf.float32, vsi_initializer)
        Wz = tf.get_variable('Wz', (i_sz, o_sz), tf.float32, vsi_initializer)
        W = tf.get_variable('W', (i_sz, o_sz), tf.float32, vsi_initializer)
        br = tf.get_variable('br', o_sz, tf.float32, one_initializer)
        bz = tf.get_variable('bz', o_sz, tf.float32, one_initializer)
        b = tf.get_variable('b', o_sz, tf.float32, zero_initializer)
        h_init = tf.get_variable('h_init', (batch_size, o_sz), tf.float32, zero_initializer)
    
    return h_init

def cell(x, h_1):
    """Hand-coded GRU"""
    with tf.variable_scope('GRU', reuse=True):
        Wr = tf.get_variable('Wr')
        Wz = tf.get_variable('Wz')
        W = tf.get_variable('W')
        br = tf.get_variable('br')
        bz = tf.get_variable('bz')
        b = tf.get_variable('b')
    
    xh = tf.concat([x, h_1], axis=1)
    r = tf.sigmoid(tf.matmul(xh, Wr) + br)     # Eq. 5
    rh_1 = r * h_1
    xrh_1 = tf.concat([x, rh_1], axis=1)
    z = tf.sigmoid(tf.matmul(xh, Wz) + bz)     # Eq. 6
    h_tild = tf.tanh(tf.matmul(xrh_1, W) + b)  # Eq. 8
    h = z*h_1 + (1-z)*h_tild                   # Eq. 7
    
    return h

def model(config):
    '''Embedding layer, RNN and hidden layer'''
    with tf.name_scope('embedding'):
        x = tf.placeholder(tf.int32, shape=(config.batch_size, config.num_rnn_steps), name='input')
        with tf.variable_scope('embedding', initializer=rui_initializer):
            embed_w = tf.get_variable('w', [config.vocab_size, config.embed_size])
        embed_out = tf.nn.embedding_lookup(embed_w, x, name='output')
        tf.summary.histogram('embed_out', embed_out)  # for TensorBoard
        # keep only top N=embed_vis_depth vectors for TensorBoard visualization:
        top_embed = tf.Variable(tf.zeros([config.embed_vis_depth, config.embed_size],
                                         dtype=tf.float32),
                                name="top_n_embedding")
        assign_embed = top_embed.assign(embed_w[:config.embed_vis_depth, :])
    
    s = [init_rnn_cell(embed_out[:, 0, :], config.rnn_size, config.batch_size)]
    for i in range(config.num_rnn_steps):
        s_1 = s[-1]
        s.append(cell(embed_out[:, i, :], s_1))
        
    with tf.name_scope('hidden'):
        rnn_last_output = s[-1]
        with tf.variable_scope('hidden'):
            hid_w = tf.get_variable('w', (config.rnn_size, config.hidden_size),
                                   initializer=vsi_initializer)
            hid_b = tf.get_variable('b', config.hidden_size, initializer=zero_initializer)
        hid_out = tf.nn.tanh(tf.matmul(rnn_last_output, hid_w) + hid_b)
        tf.summary.histogram('hid_out', hid_out)  # for TensorBoard
            
    return hid_out, x, top_embed, assign_embed, embed_w

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
        top_embed = tf.Variable(tf.zeros([config.embed_vis_depth, config.hidden_size],
                                         dtype=tf.float32),
                                name="top_n_embedding")
        assign_embed = top_embed.assign(w[:config.embed_vis_depth, :])
    
    with tf.name_scope('predict'):
        y_hat = tf.argmax(tf.matmul(hid_out, w, transpose_b=True) + b, axis=1)
    
    return y, batch_loss, y_hat, top_embed, assign_embed, w

def train(config, batch_loss):
    with tf.name_scope('optimize'):
        step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.MomentumOptimizer(config.learn_rate, config.momentum)
        train_op = optimizer.minimize(batch_loss, name='minimize_op', global_step=step)
    
    return train_op, step

class MyGraph(object):
    def __init__(self, config):
        self.hid_out, self.x, self.top_embed_in, self.assign_embed_in, self.embed_w = model(config)
        self.y, self.batch_loss, self.y_hat, self.top_embed_out, self.assign_embed_out, self.w =             loss(config, self.hid_out)
        self.train_op, self.step = train(config, self.batch_loss)
        self.init = tf.global_variables_initializer()
        # Save histogram of all trainable variables for viewing in TensorBoard
        [tf.summary.histogram(v.name.replace(':', '_'), v) for v in tf.trainable_variables()]
        self.summ = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=2)       

def embed_vis(summary_writer, g):
    """Setup for Tensorboard embedding visualization"""
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    # input embedding
    embedding = config.embeddings.add()
    embedding.tensor_name = g.top_embed_in.name  
    embedding.metadata_path = 'embed_metadata.tsv'
    # output embedding
    embedding = config.embeddings.add()
    embedding.tensor_name = g.top_embed_out.name
    embedding.metadata_path = 'embed_metadata.tsv'
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)  

def build_logfile_name(config):
    """Generate logfile name based on training configuration and model params"""
    logfile_name = ('../tf_logs/st={}_es={}_rs={}_lr={}_e={}'.
                    format(config.num_rnn_steps, 
                           config.embed_size, config.rnn_size,
                           config.learn_rate, config.epochs))
    
    return logfile_name

# Train
logfile_name = build_logfile_name(config)
summary_interval = 250
move_avg_len = 20  # number of batches to average loss over
move_avg_loss = np.zeros(move_avg_len)
with tf.Graph().as_default():
    g = MyGraph(config)
    with tf.Session() as sess:
        sess.run(g.init)
        writer = tf.summary.FileWriter(logfile_name+'/', tf.get_default_graph())
        start_time = time.time()
        for e in range(config.epochs):
            for t in tqdm_notebook(feeder(config, word_array),
                                   total=epoch_len(config, word_array),
                                   desc='Epoch #{}'.format(e+1), leave=False,
                                  mininterval=1):
                feed = {g.x: t[0], g.y: t[1]}
                [_, batch_loss, step] = sess.run([g.train_op, g.batch_loss, g.step],
                                               feed_dict=feed)
                move_avg_loss[step % move_avg_len] = batch_loss
                if (step % summary_interval) == 0:
                    sess.run([g.assign_embed_in, g.assign_embed_out])
                    writer.add_summary(sess.run(g.summ, feed_dict=feed), step)
            print('Epoch #{} Loss ({} batch average): {}'.
                  format(e+1, move_avg_len, np.mean(move_avg_loss)))
            last_saved = g.saver.save(sess, logfile_name, global_step=e)
        print("--- %s seconds ---" % (time.time() - start_time))            
        embed_vis(writer, g)
        writer.close()
        
# Write metadata file for TensorBoard embedding visualization
with open('../tf_logs/embed_metadata.tsv', 'w') as f:
    for i in range(config.embed_vis_depth):
        f.write(reverse_dict[i]+'\n')   

help(tqdm_notebook)

import tqdm

help(tqdm._tqdm_notebook.tqdm_notebook)



