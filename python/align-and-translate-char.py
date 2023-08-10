import tensorflow as tf
import numpy as np

import jieba

import pickle
with open('align_and_translate_char_50000/dic.pkl','rb') as fhdl:
    ( 
        ind2ch,
        ch2ind,
        ind2en,
        en2ind,
    ) = pickle.load(fhdl)

src_vocab_size = 50003#len(ind2en) + 3
target_vocat_size = 8826#len(ind2ch) + 3
attention_hidden_size = 512
attention_output_size = 512
embedding_size = 512
seq_max_len = 40
num_units = 512
batch_size = 4#64
layer_number = 2
max_grad = 1.0
dropout = 0.2

from tensorflow.python.layers import core as layers_core

import tensorflow as tf
import tflearn
tf.reset_default_graph()
config = tf.ConfigProto(log_device_placement=True,allow_soft_placement = True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)


with tf.device('/gpu:1'):
    initializer = tf.random_uniform_initializer(
        -0.08, 0.08)
    tf.get_variable_scope().set_initializer(initializer)
    
    x = tf.placeholder("int32", [None, None])
    y = tf.placeholder("int32", [None, None])
    y_in = tf.placeholder("int32",[None,None])
    x_len = tf.placeholder("int32",[None])
    y_len = tf.placeholder("int32",[None])
    x_real_len = tf.placeholder("int32",[None])
    y_real_len = tf.placeholder("int32",[None])
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    # embedding
    embedding_encoder = tf.get_variable(
        "embedding_encoder", [src_vocab_size, embedding_size],dtype=tf.float32)
    embedding_decoder = tf.get_variable(
        "embedding_decoder", [target_vocat_size, embedding_size],dtype=tf.float32)
    
    encoder_emb_inp = tf.nn.embedding_lookup(
        embedding_encoder, x)
    decoder_emb_inp = tf.nn.embedding_lookup(
        embedding_decoder, y_in)
    
    # encoder
    num_bi_layers = int(layer_number / 2)
    cell_list = []
    for i in range(num_bi_layers):
        cell_list.append(
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(num_units), input_keep_prob=(1.0 - dropout)
            )
        )
    if len(cell_list) == 1:
        encoder_cell = cell_list[0]
    else:
        encoder_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
        
    cell_list = []
    
    for i in range(num_bi_layers):
        cell_list.append(
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(num_units), input_keep_prob=(1.0 - dropout)
            )
        )
    if len(cell_list) == 1:
        encoder_backword_cell = cell_list[0]
    else:
        encoder_backword_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    
    bi_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
        encoder_cell,encoder_backword_cell, encoder_emb_inp,
        sequence_length=x_len,dtype=tf.float32)
    encoder_outputs = tf.concat(bi_outputs, -1)
    
    if num_bi_layers == 1:
        encoder_state = bi_encoder_state
    else:
        encoder_state = []
        for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
        encoder_state = tuple(encoder_state)
    
    # decoder 
    #decoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
    cell_list = []
    for i in range(layer_number):
        cell_list.append(
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(num_units), input_keep_prob=(1.0 - dropout)
            )
        )
    if len(cell_list) == 1:
        decoder_cell = cell_list[0]
    else:
        decoder_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    
    # Helper
    
    # attention
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        attention_hidden_size, encoder_outputs,
        memory_sequence_length=x_real_len,scale=True)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell, attention_mechanism,
        attention_layer_size=attention_output_size)
    
    
    projection_layer = layers_core.Dense(
        target_vocat_size, use_bias=False)
    
    
    
    # Dynamic decoding
    with tf.variable_scope("decode_layer"):
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp,sequence_length= y_len)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state = decoder_cell.zero_state(dtype=tf.float32,batch_size=batch_size),
            output_layer=projection_layer)
       
        outputs, _,___  = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output

        target_weights = tf.sequence_mask(
            y_real_len, seq_max_len, dtype=logits.dtype)
    
    # predicting
    # Helper
    with tf.variable_scope("decode_layer", reuse=True):
        helper_predict = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding_decoder,
            tf.fill([batch_size], ch2ind['<go>']), 0)
        decoder_predict = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper_predict, initial_state = decoder_cell.zero_state(dtype=tf.float32,batch_size=batch_size),
            output_layer=projection_layer)
        outputs_predict,_, __ = tf.contrib.seq2seq.dynamic_decode(
            decoder_predict, maximum_iterations=seq_max_len * 2)
    translations = outputs_predict.sample_id

    # calculate loss
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    train_loss = (tf.reduce_sum(crossent * target_weights) /
        batch_size)
    
    optimizer_ori = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, max_grad)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = optimizer_ori.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=global_step)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(train_loss)
    #trainop = tflearn.TrainOp(loss=train_loss, optimizer=optimizer,
    #                          metric=train_loss, batch_size=64)

saver = tf.train.Saver()

saver.restore(session,'align_and_translate_char_50000/align_and_translate_model')

sent = "You mother fucker , i will kill you someday ."

sent = 'However, they do not fall on to hard ground but on to empty cardboard boxes covered with a mattress.'

sent = """On the substance, it was a performance that quickly emboldened white nationalist groups and appeared certain to heighten racial tensions and fear in the country."""

sent = " Teaming up with Chinese search engine Baidu, online travel agency Ctrip has launched a new translation service for Chinese tourists abroad."

sent = [i.lower() for i in jieba.cut(sent) if i.strip()]

sents = [en2ind.get(i,en2ind['<unk>']) for i in sent]

sents = tf.contrib.keras.preprocessing.sequence.pad_sequences([sents],seq_max_len,padding='post')

tran = session.run([translations],feed_dict={x:np.repeat(sents,4,axis=0),x_len:[35] * 4, x_real_len:[sum(sents[0] > 0) + 1] * 4})

from collections import Counter
for i,j in  Counter(''.join([ind2ch.get(i,'') for i in j]) for j in tran[0]).most_common(5):
    print i,j

import re

regx = re.compile('(?<![\d.])(?!\.\.)'
                  '(?<![\d.][eE][+-])(?<![\d.][eE])(?<!\d[.,])'
                  '' #---------------------------------
                  '([+-]?)'
                  '(?![\d,]*?\.[\d,]*?\.[\d,]*?)'
                  '(?:0|,(?=0)|(?<!\d),)*'
                  '(?:'
                  '((?:\d(?!\.[1-9])|,(?=\d))+)[.,]?'
                  '|\.(0)'
                  '|((?<!\.)\.\d+?)'
                  '|([\d,]+\.\d+?))'
                  '0*'
                  '' #---------------------------------
                  '(?:'
                  '([eE][+-]?)(?:0|,(?=0))*'
                  '(?:'
                  '(?!0+(?=\D|\Z))((?:\d(?!\.[1-9])|,(?=\d))+)[.,]?'
                  '|((?<!\.)\.(?!0+(?=\D|\Z))\d+?)'
                  '|([\d,]+\.(?!0+(?=\D|\Z))\d+?))'
                  '0*'
                  ')?'
                  '' #---------------------------------
                  '(?![.,]?\d)')



simpler_regex = re.compile('(?<![\d.])0*(?:'
                           '(\d+)\.?|\.(0)'
                           '|(\.\d+?)|(\d+\.\d+?)'
                           ')0*(?![\d.])')


def split_outnumb(string, regx=regx, a=0):
    excluded_pos = [x for mat in regx.finditer(string) for x in range(*mat.span()) if string[x]=='.']
    li = []
    for xdot in (x for x,c in enumerate(string) if c=='.' and x not in excluded_pos):
        li.append(string[a:xdot])
        a = xdot + 1
    li.append(string[a:])
    return li

sent = """On the substance, it was a performance that quickly emboldened white nationalist groups.And appeared certain
to heighten racial tensions and fear in the country."""
sent = filter(lambda x:x != "\n" and x != "\t",sent)
split_outnumb(sent)



