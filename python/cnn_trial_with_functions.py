# !sudo pip install -U nltk
# !sudo pip install wget
# !sudo pip install tflearn

import sys
print(sys.version)
import tensorflow as tf
print(tf.__version__)
import nltk

import pandas as pd
import gzip
import time
import tensorflow as tf
import tflearn
# Install a few python packages using pip
#from common import utils
from w266_common import utils, vocabulary
utils.require_package('nltk')
utils.require_package("wget")      # for fetching dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Standard python helper libraries.
from __future__ import print_function
from __future__ import division
import os, sys, time
import collections
import itertools

# Numerical manipulation libraries.
import numpy as np
from scipy import stats, optimize

import nltk
nltk.download('punkt')
from nltk import word_tokenize


#comment or uncomment based on anamika/ arunima
# Helper libraries
#from common import utils, vocabulary, glove_helper


from w266_common import glove_helper



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

#Using pretrained GLove embeddings
hands = glove_helper.Hands(ndim=100)  # 50, 100, 200, 300 dim are available
hands.shape
print(hands.shape[1])

df_toys = getDF('reviews_Toys_and_Games.json.gz')

df_vid = getDF('reviews_Video_Games.json.gz')

df_aut = getDF('reviews_Automotive.json.gz')

df_hnk = getDF('reviews_Home_and_Kitchen.json.gz')

df_hnk.to_pickle('./df.hnk.pkl')

#Looking at a few examples of review text
# print('Toys reviews examples\n')
# for i in range(1):
#     print(df_toys['reviewerID'].iloc[i])
#     print(df_toys['reviewText'].iloc[i])

# print('\n Video games reviews examples\n')
# for i in range(1):
#     print(df_vid['reviewerID'].iloc[i])
#     print(df_vid['reviewText'].iloc[i])
    
# print('\n Automobile reviews examples\n')
# for i in range(1):
#     print(df_aut['reviewerID'].iloc[i])
#     print(df_aut['reviewText'].iloc[i])
    
print('\n Home and Kitchen reviews examples\n')
for i in range(2):
    print(df_hnk['reviewerID'].iloc[i])
    print(df_hnk['reviewText'].iloc[i])

# Create train,dev,test split
from sklearn.model_selection import train_test_split
train_toys,devtest = train_test_split(df_toys, test_size=0.4, random_state=42)
dev_toys,test_toys = train_test_split(devtest,test_size = 0.5, random_state=42)
print('Toy reviews train, dev and test set dataframe shape:',train_toys.shape,dev_toys.shape,test_toys.shape)

#For Video games reviews
train_vid,devtest = train_test_split(df_vid, test_size=0.4)
dev_vid,test_vid = train_test_split(devtest,test_size = 0.5)
print('Video games reviews train, dev and test set dataframe shape:',train_vid.shape,dev_vid.shape,test_vid.shape)

#For Auto reviews
train_aut,devtest = train_test_split(df_aut, test_size=0.4)
dev_aut,test_aut = train_test_split(devtest,test_size = 0.5)
print('Auto reviews train, dev and test set dataframe shape:',train_aut.shape,dev_aut.shape,test_aut.shape)

#For Home and Kitchen reviews
train_hnk,devtest = train_test_split(df_hnk, test_size=0.4)
dev_hnk,test_hnk = train_test_split(devtest,test_size = 0.5)
print('Home and Kitchen reviews train, dev and test set dataframe shape:',train_hnk.shape,dev_hnk.shape,test_hnk.shape)

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

list_df = ['toys','vid','aut','hnk'] #list of keys that refer to each dataframe. Adding a new dataframe would require updating this list
#list_df = ['toys','hnk']
dict_train_df = {} #Dict to store train input data frame for each domain, can be accessed by using domain name as key
dict_dev_df = {} #Dict to store dev input data frame for each domain, can be accessed by using domain name as key
dict_train_y = {} #Dict to store binarized train data label for each domain
dict_dev_y = {} #Dict to store binarized dev data label for each domain
#print(len(dict_train_df))

def create_sized_data(size = 100000):
    size_train = size #Set size of train set here. This is a hyperparameter.
    key = list_df[0]
    #print('Toys reviews\n')
    dict_train_df[key], dict_dev_df[key], dict_train_y[key], dict_dev_y[key] = set_df_size(size_train,train_toys,dev_toys)
    #print('\n Video games reviews\n')
    
    
    key = list_df[1]
    dict_train_df[key], dict_dev_df[key], dict_train_y[key], dict_dev_y[key] = set_df_size(size_train,train_vid,dev_vid)
    #print('\n Auto reviews\n')
    key = list_df[2]
    dict_train_df[key], dict_dev_df[key], dict_train_y[key], dict_dev_y[key] = set_df_size(size_train,train_aut,dev_aut)
    #print('\n Home and Kitchen reviews\n')
    
   
    key = list_df[3]
    dict_train_df[key], dict_dev_df[key], dict_train_y[key], dict_dev_y[key] = set_df_size(size_train,train_hnk,dev_hnk)
    
create_sized_data(100000)
#print(len(dict_train_df))

# Hyperparameters
max_length = 150

vocab_processor = tflearn.data_utils.VocabularyProcessor(max_length, min_frequency=0)
#Note : Above function was used instead of the below, which is deprecated. 
# vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_length)

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

# x_train, y_train, x_dev, y_dev, V = process_inputs('hnk',vocab_processor)

# #Print a few examples for viewing
# print('sample review',dict_train_df['hnk']['reviewText'].iloc[3],'\n')
# print('corresponding ids\n',x_train[3])
# print('sample review',dict_dev_df['hnk']['reviewText'].iloc[3],'\n')
# print('corresponding ids\n',x_dev[3])

#Converting reviews to ids for all domains and add padding.

#dict_vectorizers = {} #Dict to store the padded tokens  on each domain
dict_train_ids = {} #Dict to store train data reviews as sparse matrix of word ids
dict_dev_ids = {} #Dict to store dev data reviews as sparse matrix of word ids
dict_cnn = {} #Dict to store cnn model developed on each domain. Assumes input features are developed using the corresponding count_vectorizer
dict_dev_ypred = {} #Dict to store dev predictions
dict_vocab_len = {} #Store vocab length of each domain
for key in list_df:
    
    #Converting ratings to tokenized word id counts as a sparse matrix using count_vectorizer
    #dict_vectorizers[key] = CountVectorizer()
    print(key)
    dict_train_ids[key], dict_train_y[key],dict_dev_ids[key], dict_dev_ypred[key], dict_vocab_len[key] = process_inputs(key,vocab_processor)
    
#     dict_train_ids[key] = dict_vectorizers[key].fit_transform(dict_train_df[key].reviewText)
#     dict_dev_ids[key] = dict_vectorizers[key].transform(dict_dev_df[key].reviewText)
    print("Number words in training corpus for",key,(dict_vocab_len[key]))
    print(key,'dataset id shapes',dict_train_ids[key].shape, dict_dev_ids[key].shape)
    
    
    
#     #Building a Naive Bayes model to predict the ratings
#     dict_nb[key] = MultinomialNB()
#     dict_nb[key].fit(dict_train_ids[key],dict_train_y[key])
#     dict_dev_ypred[key] = dict_nb[key].predict(dict_dev_ids[key])
#     acc = accuracy_score(dict_dev_y[key], dict_dev_ypred[key])
#     print("Accuracy on",key,"dev set for binary prediction with toys naive bayes model: {:.02%}".format(acc))
#     print('Corresponding classification report\n',classification_report(dict_dev_y[key], dict_dev_ypred[key]))

#Print a few examples for viewing
print('sample review',dict_train_df['toys'].reviewText.iloc[3],'\n')
print('corresponding ids\n',dict_train_ids['toys'][3],'\n')
# print('sample review',dict_dev_ids['toys']['reviewText'].iloc[3],'\n')
# print('corresponding ids\n',x_dev[3])

# print(y_train.shape)
# print(y_dev.shape)
# print(np.mean(y_train))

class TextCNN(object):

    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__( self, sequence_length, num_classes, vocab_size, learning_rate, momentum, embedding_size, 
                 gl_embed, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer

        with tf.name_scope("embedding"):
            #self.W = tf.get_variable("W_in",[vocab_size, embedding_size],initializer =tf.random_uniform_initializer(0,1)) #from wildML
            self.W=tf.get_variable(name="embedding_",shape=gl_embed.shape,
                                       initializer=tf.constant_initializer(gl_embed),trainable=True)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            #print('embedded_chars',self.embedded_chars.get_shape())
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            #print('embedded_chars_expanded',self.embedded_chars_expanded.get_shape())

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                Wname = "w_%d"%filter_size
                W = tf.get_variable(Wname, shape = filter_shape, initializer = tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d( self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")

                # Apply nonlinearity
                conv+= b
                h = tf.nn.relu(conv, name="relu")
                #print('h',h.get_shape())

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                    padding='VALID', name="pool")
                #print('pooled',pooled.get_shape())
                pooled_outputs.append(pooled)
                #print('pooled_outputs',type(pooled_outputs))
                #print('pooled_outputs as array',type(np.array(pooled_outputs)),np.array(pooled_outputs).shape)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        #print('h_pool',self.h_pool.get_shape())
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        #print('h_pool_flat',self.h_pool_flat.get_shape())
        
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_filters_total, num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.0, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            #print('self.scores',self.scores.get_shape())
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            #print('self.predictions',self.predictions.get_shape())
            
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            #self.loss = tf.losses.mean_squared_error(self.input_y, self.scores)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
#             correct_pred = tf.equal(tf.cast(tf.round(self.scores), tf.int32), self.input_y)
#             self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
        with tf.name_scope('train'):
            #self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate,momentum=momentum,use_nesterov=True).minimize(self.loss)

def batch_generator(ids, labels, batch_size=100):
            #ids is input, X_train
            #need to fix this to shuffle between epochs
    
            n_batches = len(ids)//batch_size
            ids, labels = ids[:n_batches*batch_size], labels[:n_batches*batch_size]
            shuffle = np.random.permutation(np.arange(n_batches*batch_size))
            ids, labels = ids[shuffle], labels[shuffle]

    
            for ii in range(0, len(ids), batch_size):
                yield ids[ii:ii+batch_size], labels[ii:ii+batch_size]

# for ii, (x, y) in enumerate(batch_generator(x_train, y_train, batch_size), 1):
#     print(ii,type(x),x.shape,type(np.array(y)),y.shape)

#Model parameters

#embed_dim = 50 #use when not using pre-trained embeddings
embed_dim = hands.shape[1]
filter_sizes= [3,4,5]
num_filters = 256
l2_reg_lambda = 0
learning_rate = 0.01
momentum = 0.9
keep_prob = 0.8
evaluate_train = 3 # of epochs at which to print test accuracy
evaluate_dev = 6 # of epochs at which to estimate and print dev accuracy
time_print = 12 # of epochs at which to print time taken
num_classes = 2
num_epochs = 60
num_checkpoints = 1
batch_size = 64
#batch_size = 128
#batch_size = 32

out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "cnn"))
print("Writing to {}\n".format(out_dir))

#Actual training loop:

def train_cnn(key):
    
    
    x_train = dict_train_ids[key]
    y_train = dict_train_y[key]
    x_dev = dict_dev_ids[key]
    y_dev = dict_dev_ypred[key]
    V = dict_vocab_len[key]
    
    with tf.Graph().as_default():

        with tf.Session() as sess:
        
            cnn = TextCNN(sequence_length=x_train.shape[1], num_classes=num_classes, vocab_size=V, learning_rate = learning_rate,
                        momentum = momentum, embedding_size=embed_dim, gl_embed = hands.W, filter_sizes= filter_sizes, 
                      num_filters=num_filters, l2_reg_lambda=l2_reg_lambda)
            sess.run(tf.global_variables_initializer())
            print('completed cnn creation')

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

            # Write vocabulary
            ## vocab_processor.save(os.path.join(out_dir, "vocab"))

            print('# batches =', len(x_train)//batch_size)
            start = time.time()
            for e in range(num_epochs):
                    
                #sum_scores = np.zeros((batch_size*(len(x_train)//batch_size),1))
                total_loss = 0
                total_acc = 0
                for i, (x, y) in enumerate(batch_generator(x_train, y_train, batch_size), 1):
                    feed = {cnn.input_x: x, cnn.input_y: y, cnn.dropout_keep_prob: keep_prob}
                    _, scores, loss, accuracy = sess.run([cnn.optimizer,cnn.scores,cnn.loss, cnn.accuracy],feed_dict = feed)
                    total_loss += loss*len(x)
                    total_acc += accuracy*len(x)
                
                    #sum_scores[i*batch_size:(i+1)*batch_size,:] = scores
                    #print(np.mean(sum_scores))
                    #time_str = datetime.datetime.now().isoformat()
                if e%evaluate_train==0:
                    avg_loss = total_loss/(batch_size*(len(x_train)//batch_size))
                    avg_acc = total_acc/(batch_size*(len(x_train)//batch_size))
                    print("Train epoch {}, average loss {:g}, average accuracy {:g},".format(e, avg_loss, avg_acc))
                    #print('average',np.mean(scores))

                if e%evaluate_dev==0:
                    total_loss = 0
                    total_acc = 0
                    num_batches = 0
                    for ii, (x, y) in enumerate(batch_generator(x_dev, y_dev, batch_size), 1):
                        feed_dict = {cnn.input_x: x, cnn.input_y: y, cnn.dropout_keep_prob: 1.0}
                        loss, accuracy = sess.run([cnn.loss, cnn.accuracy],feed_dict)
                        total_loss += loss*len(x)
                        total_acc += accuracy*len(x)
                        num_batches += 1
                    avg_loss = total_loss/(num_batches*batch_size)
                    avg_acc = total_acc/(num_batches*batch_size)
                #time_str = datetime.datetime.now().isoformat()
                    print("\t\tDev epoch {}, average loss {:g}, average accuracy {:g},".format(e, avg_loss, avg_acc))
                if e%time_print == 0:
                    end = time.time()
                    print("\t\t\t\t    Time taken for",e,"epochs = ", end-start)

for key in list_df:
    print(key)
    train_cnn(key)
    

