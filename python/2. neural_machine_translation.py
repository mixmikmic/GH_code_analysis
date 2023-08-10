from __future__ import division

from utils import preprocess, decode_sequence
from ops import lstm_cell, rnn_encoder, attention_mechanism
from ops import rnn_decoder, rnn_decoder_test
import tensorflow as tf
import numpy as np

source_sequences = open('data/source.txt').readlines()
target_sequences = open('data/target.txt').readlines()

k = 3
for eng, kor in zip(source_sequences[0:k], target_sequences[0:k]):
    print eng, kor

source, source_mask, word_to_idx_src, source_seq_length, source_vocab_size = preprocess(source_sequences)
target, target_mask, word_to_idx_trg, target_seq_length, target_vocab_size = preprocess(target_sequences)

eng = decode_sequence(source[0:1], word_to_idx_src)
kor =  decode_sequence(target[0:1], word_to_idx_trg)

for e, k in zip(eng, kor):
    print e
    print k

batch_size = 2
dim_h = 128
dim_emb = 64 

params = {}

# weights and biases for rnn encoder 
params['w_encoder'] = tf.Variable(tf.random_normal(shape=[dim_h + dim_emb, dim_h*4], stddev=0.1), name='w_encoder')
params['b_encoder'] = tf.Variable(tf.zeros(shape=[dim_h*4]), name='b_encoder')

# weights and biases for rnn decoder
params['w_decoder'] = tf.Variable(tf.random_normal(shape=[dim_h*2 + dim_emb, dim_h*4], stddev=0.1), name='w_decoder')
params['b_decoder'] = tf.Variable(tf.zeros(shape=[dim_h*4]), name='w_decoder')

# weigths and biases for attention mechanism
params['w1_att'] = tf.Variable(tf.random_normal(shape=[dim_h, dim_h], stddev=0.1), name='w1_att')
params['w2_att'] = tf.Variable(tf.random_normal(shape=[dim_h, dim_h], stddev=0.1), name='w2_att')
params['b_att'] = tf.Variable(tf.zeros(shape=[dim_h]), name='b_att')
params['w3_att'] = tf.Variable(tf.random_normal(shape=[dim_h, 1], stddev=0.1), name='w3_att')

# embedding matrices for source and target languages
params['w_emb_src'] = tf.Variable(tf.random_uniform(shape=[source_vocab_size, dim_emb], minval=-1.0, maxval=1.0),
                                 name='w_emb_src')
params['w_emb_trg'] = tf.Variable(tf.random_uniform(shape=[target_vocab_size, dim_emb], minval=-1.0, maxval=1.0),
                                 name='w_emb_trg')


# weigths and biases for initializing initial cell and hidden state in decoder
params['w_init_c'] = tf.Variable(tf.random_normal(shape=[dim_h, dim_h], stddev=0.1), name='w_init_c')
params['b_init_c'] = tf.Variable(tf.zeros(shape=[dim_h]), name='b_init_c')
params['w_init_h'] = tf.Variable(tf.random_normal(shape=[dim_h, dim_h], stddev=0.1), name='w_init_h')
params['b_init_h'] = tf.Variable(tf.zeros(shape=[dim_h]), name='b_init_h')

# weights and biases for computing logits (include softmax layer)
params['w1_logit'] = tf.Variable(tf.random_normal(shape=[dim_h, dim_h], stddev=0.1), name='w1_logit')
params['b1_logit'] = tf.Variable(tf.zeros(shape=[dim_h]), name='b1_logit')
params['w2_logit'] = tf.Variable(tf.random_normal(shape=[dim_h, target_vocab_size], stddev=0.1), name='w2_logit')
params['b2_logit'] = tf.Variable(tf.zeros(shape=[target_vocab_size]), name='b2_logit')

tf_source = tf.placeholder(dtype=tf.int64, shape=[None, source_seq_length], name='source_seq')
tf_source_mask = tf.placeholder(dtype=tf.int64, shape=[None, source_seq_length], name='source_mask')
tf_target = tf.placeholder(dtype=tf.int64, shape=[None, target_seq_length], name='target_seq')
tf_target_mask = tf.placeholder(dtype=tf.int64, shape=[None, target_seq_length], name='target_mask')

# encoder
h_encoded = rnn_encoder(tf_source, params) # (batch_size, source_seq_length, dim_h)

# decoder (train mode)
loss = rnn_decoder(tf_target, h_encoded, tf_source_mask, tf_target_mask, params)

# decoder (test mode)
sampled_seq = rnn_decoder_test(h_encoded, tf_source_mask, word_to_idx_trg, params) # (batch_size, target_seq_length-1)

# train op
with tf.name_scope('optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    grads = tf.gradients(loss, tf.trainable_variables())
    grads_and_vars = list(zip(grads, tf.trainable_variables()))
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

# add summary op   
tf.scalar_summary('batch_loss', loss)
for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)
for grad, var in grads_and_vars:
    tf.histogram_summary(var.op.name+'/gradient', grad)

summary_op = tf.merge_all_summaries() 

num_epoch = 100
tot_batch = len(source)
num_iter_per_epoch = int(np.ceil(tot_batch / batch_size)) 
log_path = 'log/'
print_every = 10

config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
with tf.Session(config=config) as sess:
    tf.initialize_all_variables().run()
    summary_writer = tf.train.SummaryWriter(log_path, graph=tf.get_default_graph())
    for e in range(num_epoch):
        # TODO: random shuffle train dataset
        for i in range(num_iter_per_epoch):
            # get batch data
            source_batch = source[i*batch_size:(i+1)*batch_size, :]
            source_mask_batch = source_mask[i*batch_size:(i+1)*batch_size, :]
            target_batch = target[i*batch_size:(i+1)*batch_size, :]
            target_mask_batch = target_mask[i*batch_size:(i+1)*batch_size, :]
            
            # train op
            feed_dict={tf_source: source_batch, tf_source_mask: source_mask_batch, 
                       tf_target: target_batch, tf_target_mask: target_mask_batch}
            _, l = sess.run(fetches=[train_op, loss], feed_dict=feed_dict)
            
            if i % 5 == 0:
                summary = sess.run(summary_op, feed_dict)
                summary_writer.add_summary(summary, e*num_iter_per_epoch + i)
                
        # print sampled sequences
        if (e+1) % print_every == 0:
            print ("\nloss at epoch %d: %.3f" %(e+1, l))
            
            feed_dict={tf_source: source, tf_source_mask: source_mask}
            np_sampled_seq = sess.run(fetches=sampled_seq, feed_dict=feed_dict) 
            
            eng = decode_sequence(sequences=source, word_to_idx=word_to_idx_src)
            kor = decode_sequence(sequences=np_sampled_seq, word_to_idx=word_to_idx_trg)
            
            rand_idx = np.random.randint(tot_batch)
            print '원문: %s' %eng[rand_idx]
            print '번역: %s' %kor[rand_idx]



