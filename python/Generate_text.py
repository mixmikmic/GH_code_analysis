from __future__ import print_function
import numpy as np
import tensorflow as tf

import time
import os
from six.moves import cPickle

from simple_model import Model

save_dir='save' #model directory to load stored checkpointed models from
n=200 #number of words to sample
prime = 'Il ' #prime text to start the generation of text.
sample = 1 #0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces

data_dir = 'data/Artistes_et_Phalanges-David_Campion'# data directory containing input.txt
input_encoding = None # character encoding of input.txt, from https://docs.python.org/3/library/codecs.html#standard-encodings'
log_dir = 'logs'# directory containing tensorboard logs
save_dir = 'save' # directory to store checkpointed models
rnn_size = 256 # size of RNN hidden state
num_layers = 2 # number of layers in the RNN
model = 'lstm' # lstm model
batch_size = 50 # minibatch size
seq_length = 25 # RNN sequence length
num_epochs = 25 # number of epochs
save_every = 1000 # save frequency
grad_clip = 5. #clip gradients at this value
learning_rate= 0.002 #learning rate
decay_rate = 0.97 #decay rate for rmsprop
gpu_mem = 0.666 #%% of gpu memory to be allocated to this process. Default is 66.6%%
init_from = None

with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)

vocab_size = len(words)

model = Model(data_dir,input_encoding,log_dir,save_dir,rnn_size,num_layers,model,batch_size,seq_length,num_epochs,save_every,grad_clip,learning_rate,decay_rate,gpu_mem,init_from, vocab_size, True)

with tf.Session() as sess:
        #within a session, we initialize variables
        tf.global_variables_initializer().run()
        
        #then we define the Saver to retrieve the model
        saver = tf.train.Saver(tf.global_variables())
        
        #we retrieve the checkpoint of the stored model:
        ckpt = tf.train.get_checkpoint_state(save_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            #we restore the model
            saver.restore(sess, ckpt.model_checkpoint_path)
            
            #we create the results
            results = model.sample(sess, words, vocab, n, prime, sample)

print(results)



