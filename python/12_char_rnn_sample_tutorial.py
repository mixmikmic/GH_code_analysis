# Import Packages
import numpy as np
import tensorflow as tf
import collections
import argparse
import time
import os
from six.moves import cPickle

# Load chars and vocab
load_dir = "linux_kernel"
with open(os.path.join(load_dir, 'chars_vocab.pkl'), 'rb') as f:
    chars, vocab = cPickle.load(f)
vocab_size = len(vocab) 
print ("'vocab_size' is %d" % (vocab_size))

# Important RNN parameters 
rnn_size   = 128
num_layers = 2

batch_size = 1 # <= In the training phase, these were both 50
seq_length = 1

# Construct RNN model 
unitcell   = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
cell       = tf.nn.rnn_cell.MultiRNNCell([unitcell] * num_layers)
input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
istate     = cell.zero_state(batch_size, tf.float32)
# Weigths 
with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
    inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, input_data))
    inputs = [tf.squeeze(_input, [1]) for _input in inputs]
# Output
def loop(prev, _):
    prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
    return tf.nn.embedding_lookup(embedding, prev_symbol)
outputs, final_state = tf.nn.seq2seq.rnn_decoder(inputs, istate, cell
                                          , loop_function=None, scope='rnnlm')
output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])

logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
probs  = tf.nn.softmax(logits)

print ("그래프 준비됨")

# 세션 설정
session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True

# Restore RNN
sess = tf.Session(config=session_conf)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
ckpt  = tf.train.get_checkpoint_state(load_dir)

print (ckpt.model_checkpoint_path)
saver.restore(sess, ckpt.model_checkpoint_path)

# Sampling function
def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))

# Sample using RNN and prime characters
prime = "/* "
state = sess.run(cell.zero_state(1, tf.float32))
for char in prime[:-1]:
    x = np.zeros((1, 1))
    x[0, 0] = vocab[char]
    state = sess.run(final_state, feed_dict={input_data: x, istate:state})

# Sample 'num' characters
ret  = prime
char = prime[-1] # <= This goes IN! 
num  = 1000
for n in range(num):
    x = np.zeros((1, 1))
    x[0, 0] = vocab[char]
    [probsval, state] = sess.run([probs, final_state]
        , feed_dict={input_data: x, istate:state})
    p      = probsval[0] 
    
    sample = weighted_pick(p)
    # sample = np.argmax(p)
    
    pred   = chars[sample]
    ret    = ret + pred
    char   = pred
    
print ("샘플링 완료. \n___________________________________________\n")

print (ret)



