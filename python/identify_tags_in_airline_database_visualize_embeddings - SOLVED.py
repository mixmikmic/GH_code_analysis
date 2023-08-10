from __future__ import print_function

import os 
import sys

import numpy as np
import tensorflow as tf 

print(tf.__version__)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Read data
import pickle
import sys

atis_file = '/Users/jorge/data/training/text/atis/atis.pkl'
#atis_file = '/home/ubuntu/data/training/text/atis/atis.pkl'

with open(atis_file,'rb') as f:
    if sys.version_info.major==2:
        train, test, dicts = pickle.load(f) #python2.7
    else:
        train, test, dicts = pickle.load(f, encoding='bytes') #python3

#Dictionaries and train test partition
w2idx, ne2idx, labels2idx = dicts[b'words2idx'], dicts[b'tables2idx'], dicts[b'labels2idx']
    
idx2w  = dict((v,k) for k,v in w2idx.items())
idx2la = dict((v,k) for k,v in labels2idx.items())

train_x, _, train_label = train
test_x,  _,  test_label  = test


# Max value of word coding to assign the ID_PAD
ID_PAD = np.max([np.max(tx) for tx in train_x]) + 1
print('ID_PAD: ', ID_PAD)

def context(l, size=3):
    l = list(l)
    lpadded = size // 2 * [ID_PAD] + l + size // 2 * [ID_PAD]
    out = [lpadded[i:(i + size)] for i in range(len(l))]
    return out


# Create train and test X y.
X_trn=[]
for s in train_x:
    X_trn += context(s,size=10)
X_trn = np.array(X_trn)

X_tst=[]
for s in test_x:
    X_tst += context(s,size=10)
X_tst = np.array(X_tst)
print('X trn shape: ', X_trn.shape)
print('X_tst shape: ',X_tst.shape)

y_trn=[]
for s in train_label:
    y_trn += list(s)
y_trn = np.array(y_trn)
print('y_trn shape: ',y_trn.shape)

y_tst=[]
for s in test_label:
    y_tst += list(s)
y_tst = np.array(y_tst)
print('y_tst shape: ',y_tst.shape)

print('Num labels: ',len(set(y_trn)))
print('Num words: ',len(set(idx2w)))

#General parameters
LOG_DIR = '/tmp/tensorboard/airline/embeddings_visualize'

# data attributes
input_seq_length = X_trn.shape[1]
input_vocabulary_size = len(set(idx2w)) + 1
output_length = 127

#Model parameters
embedding_size=64
num_hidden_lstm = 128

# Save words and labels for embedding visualization
if not os.path.exists(LOG_DIR + '/train'):
    os.makedirs(LOG_DIR + '/train')
    
with open( LOG_DIR + '/train/records.tsv', "w") as record_file:
    for item in idx2w.items():
        if sys.version_info.major==2:
            record_file.write(item[1].encode('utf-8')+'\n') #python2.7
        else:
            record_file.write(item[1].decode('ascii')+'\n') #python3


from tensorflow.contrib.tensorboard.plugins import projector

# Define the tensorflow graph

graph = tf.Graph()

with graph.as_default():
    # graph definition
    # Inputs
    with tf.name_scope('Inputs') as scope:
        x = tf.placeholder(tf.int32, shape=[None, input_seq_length], name='x')
        y = tf.placeholder(tf.int64, shape=[None], name='y')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    with tf.name_scope('Embeddings') as scope:
        W_embedding = tf.Variable(tf.random_uniform([input_vocabulary_size, embedding_size], -1.0, 1.0) ,name="W_vis")
        embedding_layer = tf.nn.embedding_lookup(W_embedding, x)
        print('embedding_layer: ', embedding_layer)

    
    with tf.name_scope('RNN') as scope:
        cell_1 = tf.contrib.rnn.LSTMCell(num_hidden_lstm, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
        cell_1 = tf.contrib.rnn.DropoutWrapper(cell_1, output_keep_prob=keep_prob)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(cell_1, embedding_layer, dtype=tf.float32)
        print('lstm_outputs: ', lstm_outputs)
 

    #Dense layer form RNN outs to prediction
    with tf.name_scope('Dense') as scope:
        W_dense = tf.Variable(tf.truncated_normal([num_hidden_lstm, output_length], stddev=0.1), name='W_dense')
        b_dense = tf.Variable(tf.constant(0.1, shape=[output_length]), name='b_dense')
        dense_output = tf.nn.relu(tf.matmul(lstm_outputs[:,-1,:], W_dense) + b_dense)
        print('dense_output: ', dense_output)

        

    # Loss function
    with tf.name_scope("xent") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense_output,
                                                                       labels=y, name='cross_entropy')
        ce_summ = tf.summary.histogram("cross entropy", cross_entropy) #TENSORBOARD


    #Optimizer
    with tf.name_scope("train") as scope:
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(cross_entropy, name='train_op')


    #Accuracy
    with tf.name_scope("test") as scope:
        #Prediction
        y_pred = tf.nn.softmax(dense_output, name='y_pred')
        #Accuracy
        correct_prediction = tf.equal(tf.argmax(dense_output,1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        accuracy_summary = tf.summary.scalar("accuracy", accuracy) #TENSORBOARD

        
    # Create a saver to save weigths.
    saver = tf.train.Saver()

    
    # Merge all the summaries and write them out to /tmp/mnist_logs
    # Merge all the summaries
    with tf.name_scope('summaries') as scope:
        merged = tf.summary.merge_all()

#batch generator
def batch_generator(x=X_trn, y=y_trn, batch_size=128):
    from sklearn.utils import shuffle
    x_shuffle, y_shuffle = shuffle(x, y, random_state=0)
    for i in range(0, x.shape[0]-batch_size, batch_size):
        x_batch = x_shuffle[i:i+batch_size,:]
        y_batch = y_shuffle[i:i+batch_size]
        yield x_batch, y_batch
    
seq = batch_generator(x=X_trn, y=y_trn, batch_size=20)
print(next(seq))

# Execute the graph to train a network
import time

batch_size = 256
nEpochs = 20

start = time.time()

gpu_options = tf.GPUOptions(allow_growth = True)
with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:

    #Create sumaries writers
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', session.graph, flush_secs=2)
    test_writer  = tf.summary.FileWriter(LOG_DIR + '/test', flush_secs=2)

    
    # Define the embeddings in tensorboard
    config = projector.ProjectorConfig()
    # Create the embedding tensorboard container
    embedding = config.embeddings.add()
    # Link to the embeddings tensor
    embedding.tensor_name = W_embedding.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = LOG_DIR + '/train/records.tsv'
    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(train_writer, config)

        
    print('Initializing')
    print('Epoch - Loss(trn) -  Acc(trn)   -   Loss(tst) -   Acc(tst)')
    session.run(tf.global_variables_initializer())
    for epoch in range(nEpochs):
        ce_c=[]
        acc_c=[]
        ce_c_tst=[]
        acc_c_tst=[]
        
        batch_list = batch_generator(x=X_trn, y=y_trn, batch_size=batch_size)
        for i, batch in enumerate(batch_list):
            feedDict = {x: batch[0], y: batch[1], keep_prob: 0.5, learning_rate: 0.001} # dictionary of batch data to run the graph
            _, ce, acc = session.run([train_op, cross_entropy, accuracy], feed_dict=feedDict)
            ce_c += [ce]
            acc_c += [acc]
            
        batch_list_tst = batch_generator(x=X_tst, y=y_tst, batch_size=batch_size)
        for x_batch, y_batch in batch_list_tst:
            feedDict = {x: x_batch, y: y_batch, keep_prob: 1} # dictionary of batch data to run the graph
            ce_tst, acc_tst = session.run([cross_entropy, accuracy], feed_dict=feedDict)
            ce_c_tst += [ce_tst]
            acc_c_tst += [acc_tst]
            
        saver.save(session, LOG_DIR + "/train/model.ckpt", epoch)
        
        print(epoch, np.mean(ce_c), np.mean(acc_c), np.mean(ce_c_tst), np.mean(acc_c_tst), sep='   -   ')
print('Time to train:', time.time() - start)
# 235 secs in CPU i7 Mac
# 121 secs in GPU GT750
# 61 secs in GPU titan X



