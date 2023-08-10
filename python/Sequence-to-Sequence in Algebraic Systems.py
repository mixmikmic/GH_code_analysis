from jupyterthemes import get_themes
from jupyterthemes.stylefx import set_nb_theme
themes = get_themes()
set_nb_theme(themes[3])

import sys
sys.path.append('../')
sys.path.append('../seq2seq_regression/')
sys.path.append('../plouffe/')
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from decoder import simple_decoder_fn_train, regression_decoder_fn_inference, dynamic_rnn_decoder
import PlouffeLib
import pandas as pd
# %matplotlib inline # allows us to display matplotlib figures inline in the notebook
import matplotlib.pyplot as plt # object-oriented plotting library
'''
set parameters for the groups given by the alias
=====   =================
Alias   Property
=====   =================
'lw'    'linewidth'
'ls'    'linestyle'
'c'     'color'
'fc'    'facecolor'
'ec'    'edgecolor'
'mew'   'markeredgewidth'
'aa'    'antialiased'
=====   =================
'''
from matplotlib import rc
import numpy as np # Matrix library
from IPython.display import HTML # For video inline
import networkx as nx # imports our go to graph theory package in python
import Seq2SeqRegression
from matplotlib.animation import FuncAnimation

session = tf.InteractiveSession()

# Hyperparameters
num_frames = 200
num_nodes = 100
batch_size = 1
cell_size = 128
learning_rate = 0.005

encoder_input_ph = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_frames, num_nodes), name = 'encoder_input')
# The is_validation placeholder is used to specify whether teacher forcing is enforced or not
is_validation = tf.placeholder(tf.bool, name='is_validation')

preprocess_output = Seq2SeqRegression.preprocess(encoder_input_ph,
                                                 num_frames,
                                                 batch_size,
                                                 num_nodes)

encoder_input, seq_length, decoder_input, decoder_target = preprocess_output

encoder_output, encoder_state = Seq2SeqRegression.init_simple_encoder(LSTMCell(cell_size),
                                                                      encoder_input,
                                                                      seq_length)

context_vector = encoder_output[-1]

decoder_logits_train = Seq2SeqRegression.decoder_teacher_forcing(LSTMCell(cell_size),
                                                                 context_vector,
                                                                 encoder_state,
                                                                 decoder_target,
                                                                 seq_length,
                                                                 num_nodes)

decoder_logits_test = Seq2SeqRegression.decoder_inference(LSTMCell(cell_size),
                                                          context_vector,
                                                          encoder_state,
                                                          batch_size,
                                                          num_frames,
                                                          num_nodes)

is_teacher_forcing = tf.logical_or(Seq2SeqRegression.sample_Bernoulli(0.5), is_validation)

decoder_logits = tf.where(is_teacher_forcing, decoder_logits_train, decoder_logits_test)

loss, train_op = Seq2SeqRegression.init_optimizer(decoder_logits, 
                                                  decoder_target,
                                                  learning_rate)

decoder_prediction = Seq2SeqRegression.get_Plouffe_reconstruction(decoder_logits, 
                                                                  seq_length, 
                                                                  num_nodes)

plt.cla() # Clear figure
import PlouffeLib
N = 200 # Set number of nodes
n_frames = 200
limit = 102
G = PlouffeLib.PlouffeSequence_v2(N,98,limit,n_frames) # Initialize the graph G

anim = FuncAnimation(G.fig, G.next_frame,frames=n_frames, blit=True)
HTML(anim.to_html5_video())

plt.cla() # Clear figure
import PlouffeLib
N = 100 # Set number of nodes
n_frames = 200
limit = 102
G = PlouffeLib.PlouffeSequence_v2(N,98,limit,n_frames) # Initialize the graph G

anim = FuncAnimation(G.fig, G.next_frame,frames=n_frames, blit=True)
HTML(anim.to_html5_video())

#G = PlouffeLib.PlouffeSequence_v2(100, 0, 2, 200)
#anim = FuncAnimation(G.fig, G.next_frame, frames=200, blit=True)
#HTML(anim.to_html5_video())

Seq2SeqRegression.restore_checkpoint_variables(session, '../logs/lr2000/lr2000')

num_nodes = 100

plouffe_seq = [[[i*k%num_nodes for i in range(num_nodes)] for k in np.arange(0,20,0.1)]]

plouffe_seq = np.array(plouffe_seq)*(1/float(num_nodes))
feed_dict = {encoder_input_ph: plouffe_seq, is_validation: True}

prediction = session.run([decoder_prediction],feed_dict)

prediction[0].shape

prediction_img = np.reshape(prediction, [num_frames,num_nodes])
plt.imshow(prediction_img)
plt.show()

start_frame = [0, 20, 40, 60, 80]
next_frame = [20, 40, 60, 80, 100]
rainbows = []

for i, frame in enumerate(start_frame):
    plouffe_seq = [[[i*k%num_nodes for i in range(num_nodes)] for k in np.arange(start_frame[i],next_frame[i],0.1)]]
    plouffe_seq = np.array(plouffe_seq)*(1/float(num_nodes))
    feed_dict = {encoder_input_ph: plouffe_seq, is_validation: True}
    prediction = session.run([decoder_prediction],feed_dict)
    prediction_img = np.reshape(prediction, [num_frames,num_nodes])
    rainbows.append(prediction_img)
plt.figure(figsize=(20,20))
plt.imshow(np.concatenate(rainbows,axis=0))
plt.axis('on')
plt.savefig('predicted_rainbow_group.pdf', format='pdf', dpi=500)
plt.show()

plt.figure(figsize=(20,20))
num_nodes=100
plouffe_seq = [[[int(i*k%num_nodes) for i in range(num_nodes)] for k in np.arange(0,num_nodes,0.1)]]
plouffe_seq = np.array(plouffe_seq)*(1/float(num_nodes))
plouffe_seq[0].shape
plt.imshow(plouffe_seq[0])
plt.savefig('actual_rainbow_group.pdf', format='pdf', dpi=500)
plt.show()

# TODO: Animation for the Plouffe predicted graphs.
#Recon = PlouffeLib.ReconPlouffeViewer(prediction[0], num_frames, num_nodes)
#prediction[0]
#data = np.reshape(prediction[0], [num_frames,num_nodes]).tolist()
#data
#data[0]
#for j in range(200):
#    print [(i, int(data[j][i])) for i in range(num_nodes)] 
#(data[num_frames-1][i])) for i in range(num_nodes)]
#predicted_graph = nx.Graph(data=graph)
#plt.figure(figsize=(8,8))
#nx.draw_circular(predicted_graph, node_size=10, alpha=0.7, with_labels=False)
#plt.axis('equal')
#plt.show()
#plt.clf()
#anim = FuncAnimation(Recon.fig, Recon.next_frame, frames=num_frames-1, blit=True)
#HTML(anim.to_html5_video())
#plt.axis('equal')
#plt.show()
#plt.clf()

