# import all required libraries
import numpy as np
import tensorflow as tf

import collections
import time

# define parameters of the program
corpus_path = '../data/got_all_edited.txt'

num_epoch = 30

batch_size = 30
num_steps = 60
embedding_size = 100

hidden_unit_size = 256
vocabulary_size = 20000
learning_rate = 1e-4

STOP_TOKEN = '*STOP*'

# define a function to load and preprocess the text corpus then return list of chars
def read_file(path):
    with open(corpus_path) as f:
        char_tokens = ['*STOP*']
        text = f.read()
        char_tokens.extend(text)
        
        for i in range(len(char_tokens)):
            if char_tokens[i] == '\n':
                char_tokens[i] = STOP_TOKEN
        
        return char_tokens

def build_dataset(tokens):
    counts = []
    counts.extend(collections.Counter(tokens).most_common())
    
    dictionary = dict()
    data = list()
    
    for token, _ in counts:
        dictionary[token] = len(dictionary)
        
    for token in tokens:
        data.append(dictionary[token])
        
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    return data, dictionary, reverse_dictionary

def generate_batch(dataset, batch_size, num_steps, offset=0):
    assert offset + batch_size * num_steps < len(dataset)
    
    batch_context = np.ndarray((batch_size, num_steps), dtype=np.int32)
    batch_target = np.ndarray((batch_size, num_steps), dtype=np.int32)
    
    for i in range(batch_size):
        batch_context[i] = dataset[offset : offset+num_steps]
        batch_target[i] = dataset[offset+1 : offset+num_steps+1]
        offset += num_steps
        
    return batch_context, batch_target, offset

tokens = read_file(corpus_path)
data, tokendict, tokendictreversed = build_dataset(tokens)

vocabsize = len(tokendict)

# split the data to training set and held out set
for i in range(int(0.8*len(data)), len(data)):
    if data[i] == tokendict[STOP_TOKEN]:
        traindata = data[0:i]
        devdata = data[i:len(data)]
        break

train, label, _ = generate_batch(data, 5, num_steps)
for batch_train, batch_label in zip(train, label):
    print ''.join([tokendictreversed[token] for token in batch_train]) + ' --> '
    print ''.join([tokendictreversed[word] for word in batch_label])
    print '----------'

graph = tf.Graph()
with graph.as_default():
    # setup input and labels placeholders
    train_inputs = tf.placeholder(tf.int32, shape=[None, num_steps])
    train_labels = tf.placeholder(tf.int32, shape=[None, num_steps])
    keep_prob = tf.placeholder(tf.float32)
    bsize = tf.placeholder(tf.int32)
    
    # instantiate embedding matrix
    charvectors = tf.Variable(tf.random_normal([vocabsize, embedding_size]))
    charvector = tf.nn.embedding_lookup(charvectors, train_inputs)
    charvector = tf.nn.dropout(charvector, keep_prob)
    
    # define the rnn cell and the initial state
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_unit_size, forget_bias=0.0, state_is_tuple=True)
    init_state = rnn_cell.zero_state(bsize, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, charvector, initial_state=init_state)   
        
    # reshape the outputs into 2D vectors
    rnn_outputs = tf.reshape(rnn_outputs, [bsize * num_steps, hidden_unit_size])
     
    logits_weights = tf.Variable(tf.truncated_normal([hidden_unit_size, vocabsize], stddev=0.1))
    logits_biases = tf.Variable(tf.zeros([vocabsize]))
    logits = tf.matmul(rnn_outputs, logits_weights) + logits_biases
    
    loss_weights = tf.ones([batch_size * num_steps])
    losses = tf.nn.seq2seq.sequence_loss_by_example([logits], [train_labels], [loss_weights])
    loss = tf.reduce_sum(losses) / batch_size
        
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    
    var_saver = tf.train.Saver(tf.trainable_variables())
    path = 'checkpoints/char_rnn_langmodel.ckpt'
    
    for epoch in range(num_epoch):
        start_time = time.time()
        
        ############ train the model on the training set ######################
        offset = 0
        total_loss = 0.
        iters = 0
        seqs = 0
        training_state = sess.run(init_state, feed_dict={bsize: batch_size})
        while offset + batch_size * num_steps < len(traindata):
            batch_context, batch_target, offset = generate_batch(traindata, batch_size, num_steps, offset)
            feed_dict = {train_inputs: batch_context, train_labels: batch_target, 
                         keep_prob: .5, init_state:training_state,
                         bsize: batch_size}
            training_loss, training_state, _ = sess.run([loss, final_state, train_step], feed_dict=feed_dict)
            
            total_loss += training_loss
            iters += num_steps
        
            seqs += batch_size
        
            if seqs % 8000 == 0:
                perplexity = np.exp(total_loss / iters)
                print 'Epoch: %d, Time elapsed: %.2f s, Tokens trained: %04d, Perplexity: %.4f' %                     (epoch+1, (time.time() - start_time), offset, perplexity)
                    
        ############ evaluate the trained model on development set #############
        dev_offset = 0
        dev_total_loss = 0.
        dev_iters = 0
        dev_seqs = 0
        dev_state = sess.run(init_state, feed_dict={bsize: batch_size})
        while dev_offset + batch_size * num_steps < len(devdata):
            batch_context, batch_target, dev_offset = generate_batch(devdata, batch_size, num_steps, dev_offset)
            feed_dict = {train_inputs: batch_context, train_labels: batch_target, 
                         keep_prob: .5, init_state:dev_state,
                         bsize: batch_size}
            training_loss, dev_state = sess.run([loss, final_state], feed_dict=feed_dict)
            
            dev_total_loss += training_loss
            dev_iters += num_steps
        
            dev_seqs += batch_size
        
        perplexity = np.exp(dev_total_loss / dev_iters)
        print '*** Evaluation Epoch: %d, Tokens trained: %04d, Perplexity: %.4f ***' %             (epoch+1, dev_offset, perplexity)
                    
        # save checkpoint every 10000 train steps
        checkpoint_path = var_saver.save(sess, path)
        print 'Epoch completed. Checkpoint saved as: %s' % (checkpoint_path)

