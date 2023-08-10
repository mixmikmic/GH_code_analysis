# Standard python helper libraries.

import json, os, re, shutil, sys, time
import itertools, collections
from importlib import reload
from IPython.display import display
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# NumPy and SciPy for matrix ops
import numpy as np
import scipy.sparse
import pandas as pd
import tensorflow as tf
assert(tf.__version__.startswith("1."))
from sklearn.cross_validation import train_test_split

# NLTK for NLP utils
import nltk
import gzip
from collections import namedtuple
import tflearn
# Helper libraries
from w266_common import vocabulary, tf_embed_viz, glove_helper
from w266_common import utils; reload(utils)

#Using pretrained GLove embeddings
hands = glove_helper.Hands(ndim=100)  # 50, 100, 200, 300 dim are available

hands.shape

#Function to read the amazon review data files
def parse(path):
  print('start parse')
  start_parse = time.time()
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)
  end_parse = time.time()
  print('end parse with time for parse',end_parse - start_parse)

def getDF(path):
  print('start getDF')
  start = time.time()
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  print('end getDF')
  end = time.time()
  print('time taken to load data = ',end-start)
  return pd.DataFrame.from_dict(df, orient='index')
#df = getDF('reviews_Toys_and_Games.json.gz') #old def function corresponding to the step bt step vectorization

df_hnk = getDF('reviews_Home_and_Kitchen.json.gz')

df_vid = getDF('reviews_Video_Games.json.gz')

df_aut = getDF('reviews_Automotive.json.gz')

df_toys = getDF('reviews_Toys_and_Games.json.gz')

#Create train,dev,test split
from sklearn.model_selection import train_test_split
# train_toys,devtest = train_test_split(df_toys, test_size=0.4, random_state=42)
# dev_toys,test_toys = train_test_split(devtest,test_size = 0.5, random_state=42)
# print('Toy reviews train, dev and test set dataframe shape:',train_toys.shape,dev_toys.shape,test_toys.shape)

#For Video games reviews
# train_vid,devtest = train_test_split(df_vid, test_size=0.4)
# dev_vid,test_vid = train_test_split(devtest,test_size = 0.5)
# print('Video games reviews train, dev and test set dataframe shape:',train_vid.shape,dev_vid.shape,test_vid.shape)

#For Auto reviews
# train_aut,devtest = train_test_split(df_aut, test_size=0.4)
# dev_aut,test_aut = train_test_split(devtest,test_size = 0.5)
# print('Auto reviews train, dev and test set dataframe shape:',train_aut.shape,dev_aut.shape,test_aut.shape)

#For Home and Kitchen reviews
train_hnk,devtest = train_test_split(df_hnk, test_size=0.4, random_state=42)
dev_hnk,test_hnk = train_test_split(devtest,test_size = 0.5, random_state=42)
print('Home and Kitchen reviews train, dev and test set dataframe shape:',train_hnk.shape,dev_hnk.shape,test_hnk.shape)

#checking that we have different productids
print(train_hnk.head(5))

#Function to create a smaller sized train and dev data set. Enables testing accuracy for different sizes.
#Also binarizes the labels. Ratings of 1,2 and to 0; Ratings of 4,5 to 1.

def set_df_size(size,data_train,data_dev):
    size_train = size
    len_max_train = data_train[data_train.overall!=3].shape[0] #max possible length of train data set taking out the 3 ratings.
    #print("Number of reviews with ratings != 3 in train set",len_max_train)
    temp_size_train = min(len_max_train,size_train)

    len_max_dev = data_dev[data_dev.overall!=3].shape[0]
    #print("Number of reviews with ratings != 3 in dev set",len_max_dev)
    temp_size_dev = min(len_max_dev,int(0.3*temp_size_train)) #making the dev set about 0.3 times the train set.

    temp_train_data = data_train[data_train.overall != 3][:temp_size_train]
    #print('Size of train data',temp_train_data.shape)
    #print(temp_train_data.groupby('overall').count())
    #print(temp_train_toys[:5])

    temp_dev_data = data_dev[data_dev.overall!=3][:temp_size_dev]
    #print('Size of dev data',temp_dev_data.shape)
    #print(temp_dev_data.groupby('overall').count())
    #print(temp_dev_data[:2])
    
    #Binarize ratings
    temp_train_y = np.zeros(temp_size_train)
    temp_train_y[temp_train_data.overall > 3] = 1
    temp_dev_y = np.zeros(temp_size_dev)
    temp_dev_y[temp_dev_data.overall>3] = 1
    #print('binarized y shape',temp_train_y.shape,temp_dev_y.shape)
    #print(temp_dev_y[:20],data_dev.overall[:20])
    return temp_train_data,temp_dev_data,temp_train_y,temp_dev_y

print(print(dev_hnk.head(5)))

list_df = ['toys','vid','aut','hnk'] #list of keys that refer to each dataframe. Adding a new dataframe would require updating this list
dict_train_df = {} #Dict to store train input data frame for each domain, can be accessed by using domain name as key
dict_dev_df = {} #Dict to store dev input data frame for each domain, can be accessed by using domain name as key
dict_train_y = {} #Dict to store binarized train data label for each domain
dict_dev_y = {} #Dict to store binarized dev data label for each domain
#print(len(dict_train_df))

def create_sized_data(size = 10000):
    size_train = size #Set size of train set here. This is a hyperparameter.
#     key = list_df[0]
    #print('Toys reviews\n')
#     dict_train_df[key], dict_dev_df[key], dict_train_y[key], dict_dev_y[key] = set_df_size(size_train,train_toys,dev_toys)
#     #print('\n Video games reviews\n')
#     key = list_df[1]
#     dict_train_df[key], dict_dev_df[key], dict_train_y[key], dict_dev_y[key] = set_df_size(size_train,train_vid,dev_vid)
#     #print('\n Auto reviews\n')
#     key = list_df[2]
#     dict_train_df[key], dict_dev_df[key], dict_train_y[key], dict_dev_y[key] = set_df_size(size_train,train_aut,dev_aut)
    #print('\n Home and Kitchen reviews\n')
    key = list_df[3]
    dict_train_df[key], dict_dev_df[key], dict_train_y[key], dict_dev_y[key] = set_df_size(size_train,train_hnk,dev_hnk)
    
create_sized_data()
#print(len(dict_train_df))

list_df = ['toys','vid','aut','hnk'] #list of keys that refer to each dataframe. Adding a new dataframe would require updating this list
dict_train_df = {} #Dict to store train input data frame for each domain, can be accessed by using domain name as key
dict_dev_df = {} #Dict to store dev input data frame for each domain, can be accessed by using domain name as key
dict_train_y = {} #Dict to store binarized train data label for each domain
dict_dev_y = {} #Dict to store binarized dev data label for each domain
#print(len(dict_train_df))

def create_sized_data(size = 100000):
    size_train = size #Set size of train set here. This is a hyperparameter.
    key = list_df[3]
    #print('Toys reviews\n')

#     dict_train_df[key], dict_dev_df[key], dict_train_y[key], dict_dev_y[key] = set_df_size(size_train,train_toys,dev_toys)
#     #print('\n Video games reviews\n')
#     key = list_df[1]
#     dict_train_df[key], dict_dev_df[key], dict_train_y[key], dict_dev_y[key] = set_df_size(size_train,train_vid,dev_vid)
#     #print('\n Auto reviews\n')
#     key = list_df[2]
#     dict_train_df[key], dict_dev_df[key], dict_train_y[key], dict_dev_y[key] = set_df_size(size_train,train_aut,dev_aut)
#     #print('\n Home and Kitchen reviews\n')
    key = list_df[3]
    dict_train_df[key], dict_dev_df[key], dict_train_y[key], dict_dev_y[key] = set_df_size(size_train,train_hnk,dev_hnk)
    
create_sized_data()
#print(len(dict_train_df))



import nltk
nltk.download('punkt')
from nltk import word_tokenize

print(dict_train_df['hnk'].shape[0])

#Preprocessing steps

#Changing to nltk punkt tokenizer as the periods are not getting removed
print(dict_train_df['hnk'].shape[0])

train_cnt = collections.Counter()
x_train_tokens_list = []
start = time.time()
for i in range(dict_train_df['hnk'].shape[0]):
    #print(dict_train_df['hnk'].iloc[i][2])
    x_train_tokens = word_tokenize(dict_train_df['hnk'].iloc[i][1])
    
    

    #2. changing to lowercase and replacing numbers(are we losing any context by 
    #replacing all numbers in the review test? Are we losing any context here)
    x_tokens_canonical = utils.canonicalize_words(x_train_tokens)
    
    x_train_tokens_list.append(x_tokens_canonical)
    
    if i%10000 == 0:
        print(i) 
    #3. Build vocabulary
    for items in x_tokens_canonical:
            train_cnt[items] += 1
            
vocab = vocabulary.Vocabulary(train_cnt, size=None)  # size=None means unlimited
total_words = sum(train_cnt.values())
print("x_train_tokens_list length", len(x_train_tokens_list))
print("Vocabulary size: {:,}".format(vocab.size))
#print("Vocabulary dict: ", vocab.word_to_id)
print("Total words ",total_words )

print(len(x_train_tokens_list[1]))

print(len(x_train_tokens_list[0]))

print(x_train_tokens_list[1])

print(x_train_tokens_list[0])

#Converting all reviews to ids 
train_id_list = []
for item in x_train_tokens_list:
    train_id_list.append(vocab.words_to_ids(item))
    
test_id_list = []
for item in x_test_tokens_list:
    test_id_list.append(vocab.words_to_ids(item))    

print((x_train_tokens_list[1]))

print(max((train_id_list)))

review_lengths = [len(review) for review in x_train_tokens_list]
print("Shortest review:", min(review_lengths))
print("Longest review:",max(review_lengths))

ax = plt.axes()
sns.distplot(review_lengths)
ax.set_title("Distribution of the review lengths")
plt.plot()

pd.DataFrame(review_lengths).describe()

pd.DataFrame(review_lengths).quantile(0.9)

max_length =150
vocab_processor = tflearn.data_utils.VocabularyProcessor(max_length, min_frequency=0)
#Note : This function seems to be deprecated. Another function I ran into 
# tflearn.data_utils.VocabularyProcessor (max_document_length, min_frequency=3, vocabulary=None, tokenizer_fn=None)

def process_inputs(key, vocab_processor):
    
    # For simplicity, we call our features x and our outputs y
    start_vectorize = time.time()
    x_train = dict_train_df[key].reviewText
    y_train = dict_train_y[key]
    x_dev = dict_dev_df[key].reviewText
    y_dev = dict_dev_y[key]
    print(x_train.shape)
    
    # Train the vocab_processor from the training set
    x_train = vocab_processor.fit_transform(x_train)
    # Transform our test set with the vocabulary processor
    x_dev = vocab_processor.transform(x_dev)

    # We need these to be np.arrays instead of generators
    x_train = np.array(list(x_train))
    print(x_train.shape)
    x_dev = np.array(list(x_dev))
    y_train = np.array(y_train).astype(int)
    y_dev = np.array(y_dev).astype(int)
    
#     y_train = tf.expand_dims(y_train,1)
#     y_dev = tf.expand_dims(y_dev,1)
    print('y train shape',y_train.shape)

    V = len(vocab_processor.vocabulary_)
    print('Total words: %d' % V)
    end_vectorize = time.time()
    print('Time taken to vectorize %d size dataframe'%x_train.shape[0],end_vectorize-start_vectorize)

    # Return the transformed data and the number of words
    return x_train, y_train, x_dev, y_dev, V

x_train, y_train, x_dev, y_dev, V = process_inputs('hnk',vocab_processor)

#Print a few examples for viewing
print('sample review',dict_train_df['hnk']['reviewText'].iloc[3],'\n')
print('corresponding ids\n',x_train[3])
print('sample review',dict_dev_df['hnk']['reviewText'].iloc[3],'\n')
print('corresponding ids\n',x_dev[3])

train_ids = x_train
dev_ids = x_dev



print(train_ids.shape)

print(train_ids[0])

print(y_train.shape)
print(train_ids.shape)
print(dev_ids.shape)

lstm_size = 256
lstm_layers = 1
#batch_size = 50
batch_size = 128
learning_rate = 0.001
embed_size = 100

def batch_iterator(ids, labels, batch_size=100):
    
    n_batches = len(ids)//batch_size
    ids, labels = ids[:n_batches*batch_size], labels[:n_batches*batch_size]
    
    for ii in range(0, len(ids), batch_size):
        yield ids[ii:ii+batch_size], labels[ii:ii+batch_size]
        

def build_rnn(gl_embed=hands.W,
              embed_size=embed_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              lstm_size=lstm_size,
              lstm_layers=lstm_layers):
    
    tf.reset_default_graph()
    
    #n_words = len(vocabulary_to_int)
    
    with tf.name_scope('inputs'):
        inputs_ = tf.placeholder(tf.int32,[None, None],name='inputs_')
    with tf.name_scope('labels'):
        labels_ = tf.placeholder(tf.int32,[None, None],name='labels_')
    with tf.name_scope('keep_prob'):    
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        
    with tf.name_scope('embedding'):
#         embedding = tf.Variable(tf.random_normal((n_words,embed_size),-1,1),name='embedding_')
#         embed = tf.nn.embedding_lookup(embedding,inputs_)
        embedding=tf.get_variable(name="embedding_",shape=gl_embed.shape,
                                       initializer=tf.constant_initializer(gl_embed),trainable=False)
        embed = tf.nn.embedding_lookup(embedding, inputs_)
        
    with tf.name_scope("RNN_cells"):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=keep_prob)

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop]*lstm_layers)
        
        with tf.name_scope("RNN_init_state"):
            # Getting an initial state of all zeros
            initial_state = cell.zero_state(batch_size, tf.float32)
    
    with tf.name_scope("RNN_forward"):
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
        
    with tf.name_scope('predictions'):
        predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, 
                                                        activation_fn=tf.sigmoid,
                                                        weights_initializer=
                                                        tf.truncated_normal_initializer(stddev=0.1))   
    with tf.name_scope('cost'):
        cost = tf.losses.mean_squared_error(labels_, predictions)
        tf.summary.scalar('cost', cost)
    
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    merged = tf.summary.merge_all()
    
    # Export the nodes 
    export_nodes = ['inputs_', 'labels_','initial_state', 'final_state',
                    'keep_prob', 'cell', 'cost', 'predictions', 'optimizer',
                    'accuracy','merged']
    
    Graph = namedtuple('Graph', export_nodes)
    
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    return graph

graph = build_rnn(gl_embed=hands.W,
              embed_size=embed_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              lstm_size=lstm_size,
              lstm_layers=lstm_layers)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter('output/logs/1', sess.graph)



def train(model, epoch,train_writer,test_writer):
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        iteration = 1
        for e in range(epochs):
            state = sess.run(model.initial_state)

            for ii, (x, y) in enumerate(batch_iterator(train_ids, y_train, batch_size), 1):
                
                feed = {model.inputs_: x,
                        model.labels_: y[:, None],
                        model.keep_prob: 0.5,
                        model.initial_state: state}
                summary,loss, state, _ = sess.run([model.merged,model.cost, 
                                                   model.final_state, 
                                                   model.optimizer], feed_dict=feed)

                
                

                if iteration%100==0:
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Train loss: {:.3f}".format(loss))
                    train_writer.add_summary(summary, iteration)
                    val_acc = []
                    val_state = sess.run(model.cell.zero_state(batch_size, tf.float32))
                    
                    for x, y in batch_iterator(dev_ids, y_dev, batch_size):
                        feed = {model.inputs_: x,
                                model.labels_: y[:, None],
                                model.keep_prob: 1,
                                model.initial_state: val_state}
                        summary, dev_loss,batch_acc, val_state = sess.run([model.merged, model.cost,model.accuracy, 
                                                         model.final_state], feed_dict=feed)
                        #print('batch_acc', batch_acc)
                        val_acc.append(batch_acc)

                    test_writer.add_summary(summary,iteration)
                    print("Dev loss: {:.3f}".format(dev_loss))
                    print("Dev acc: {:.3f}".format(np.mean(val_acc)))

                iteration +=1
        saver.save(sess, "output/checkpoints/sentiment.ckpt")

print(test_ids)

lstm_size_options = [256]
lstm_layers_options = [1]
learning_rate_options = [0.001]

epochs=200
for lstm_size in lstm_size_options:
    for lstm_layers in lstm_layers_options:
        for learning_rate in learning_rate_options:
            log_string_train = 'output/logs/2/train/lr={},rl={},ru={}'.format(learning_rate, lstm_layers, lstm_size)
            log_string_test = 'output/logs/2/test/lr={},rl={},ru={}'.format(learning_rate, lstm_layers, lstm_size)
            train_writer = tf.summary.FileWriter(log_string_train)
            test_writer = tf.summary.FileWriter(log_string_test)
            
            print("lstm size: {}".format(lstm_size),
                    "nb layers : {}".format(lstm_layers),
                    "learn rate : {:.3f}".format(learning_rate))
            
            model = build_rnn(gl_embed=hands.W,
                      embed_size=embed_size,
                      batch_size=batch_size,
                      learning_rate=learning_rate,
                      lstm_size=lstm_size,
                      lstm_layers=lstm_layers)

            train(model, epochs, train_writer,test_writer)

print("run")

