import tensorflow as tf
from __future__ import print_function
import os
import numpy as np

maxlen = 25 #The maximal lenth of the sequence

import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle
import gzip

with open('zu05056_char_idx.pkl', 'rb') as f:
    if sys.version_info.major > 2:
        char_idx = pickle.load(f, encoding='latin1')
    else:
        char_idx = pickle.load(f)
        
list(char_idx.keys())[20:30],list(char_idx.values())[20:30]

# Downloading the model, if it does not exist
import urllib
import os
if not os.path.isfile('didactic_25.pb'):
    urllib.urlretrieve("https://dl.dropboxusercontent.com/u/9154523/models/rnn_fun/didactic_25.pb", "didactic_25.pb")
get_ipython().magic('ls -hl didactic_25.pb')

with tf.gfile.GFile('didactic_25.pb', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
tf.reset_default_graph()
graph = tf.Graph().as_default() 
tf.import_graph_def(graph_def,  name='')

#ops = tf.get_default_graph().get_operations()
#for i in ops:print(i.name)

graph = tf.get_default_graph()
feed = graph.get_tensor_by_name('InputData/X:0')
fetch = graph.get_tensor_by_name('FullyConnected/Softmax:0')

# The seed for prediction. Note that the seed need to be exactely of length maxlen
seed = 'Die Grundlagen war dabei '[0:maxlen]
seed, len(seed)

# Creating a one-hot-encoded matrix
X = np.zeros((1, maxlen, len(char_idx))) #One Batch, t, X_t (one-got-encoded)
for t, char in enumerate(seed):
    X[0, t, char_idx[char]] = 1.0  

with tf.Session() as sess:
    pred = sess.run(fetch, feed_dict={feed:X})

nl = np.argmax(pred) #next letter
nl

# Code taken from from tflearn
def reverse_dictionary(char_idx):
    # Build reverse dict
    rev_dic = {}
    for key in char_idx:
        rev_dic[char_idx[key]] = key
    return rev_dic

rev_dic = reverse_dictionary(char_idx)
rev_dic[nl]

def _sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    logit = np.log(a) 
    p = np.exp(logit / temperature)
    #1.001 to be on the save side, sum(p) < 1 for np.random.multinomial
    p /= (1.001 * np.sum(p))
    return np.argmax(np.random.multinomial(1, p, 1))

n = _sample(pred[0])
n, rev_dic[n]

import sys
# Code adapted from tflearn SequenceGenerator
def generate(sess, seq_seed, show=True, seq_length = 400, temperature = 0.1,  seq_maxlen=25):
    sequence = seq_seed
    generated = seq_seed
    dic = char_idx
    rev_dic = reverse_dictionary(dic)


    whole_sequence = seq_seed

    for i in range(seq_length):
        X = np.zeros((1, seq_maxlen, len(dic)))
        for t, char in enumerate(sequence):
            X[0, t, dic[char]] = 1.
        preds = sess.run(fetch, feed_dict={feed:X})[0] #Getting next letter distribution
        next_index = _sample(preds, temperature) #Sampling a letter from the distribution
        #next_index = np.argmax(preds)
        next_char = rev_dic[next_index]
        if show:
            sys.stdout.write(next_char)
            sys.stdout.flush()
        generated += next_char
        sequence = sequence[1:] + next_char
        whole_sequence += next_char
    return whole_sequence

with tf.Session() as sess:
    res = generate(sess, seed, temperature=1.0)
    print('\n')
    print(res)

ts = (1.0, 1.0, 0.5, 0.5, 0.1, 0.05)
with tf.Session() as sess:
    for t in ts:
        print()
        print("Temperature {}".format(t))
        print(generate(sess, seed, temperature=t, show=False))

