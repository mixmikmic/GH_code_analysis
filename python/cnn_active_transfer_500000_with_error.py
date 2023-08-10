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
from common import utils
utils.require_package('nltk')
utils.require_package("wget")      # for fetching dataset
#from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_curve, auc
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
# from common import utils, vocabulary, glove_helper

# from common import utils, vocabulary
from common import utils, vocabulary
from common import glove_helper

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

df_toys = getDF('/newvolume/reviews_Toys_and_Games.json.gz')
#df_toys = getDF('reviews_Toys_and_Games.json.gz')

df_vid = getDF('/newvolume/reviews_Video_Games.json.gz')
#df_vid = getDF('reviews_Video_Games.json.gz')

df_aut = getDF('/newvolume/reviews_Automotive.json.gz')
#df_aut = getDF('reviews_Automotive.json.gz')

#df_hnk = getDF('reviews_Home_and_Kitchen.json.gz')
df_hnk = getDF('/newvolume/reviews_Home_and_Kitchen.json.gz')

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
train_vid,devtest = train_test_split(df_vid, test_size=0.4, random_state=42)
dev_vid,test_vid = train_test_split(devtest,test_size = 0.5, random_state=42)
print('Video games reviews train, dev and test set dataframe shape:',train_vid.shape,dev_vid.shape,test_vid.shape)

#For Auto reviews
train_aut,devtest = train_test_split(df_aut, test_size=0.4, random_state=42)
dev_aut,test_aut = train_test_split(devtest,test_size = 0.5, random_state=42)
print('Auto reviews train, dev and test set dataframe shape:',train_aut.shape,dev_aut.shape,test_aut.shape)

#For Home and Kitchen reviews
train_hnk,devtest = train_test_split(df_hnk, test_size=0.4, random_state=42)
dev_hnk,test_hnk = train_test_split(devtest,test_size = 0.5, random_state=42)
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
dict_train_df = {} #Dict to store train input data frame for each domain, can be accessed by using domain name as key
dict_dev_df = {} #Dict to store dev input data frame for each domain, can be accessed by using domain name as key
dict_train_y = {} #Dict to store binarized train data label for each domain
dict_dev_y = {} #Dict to store binarized dev data label for each domain
#print(len(dict_train_df))

size_initial = 500000
def create_sized_data(size = 10000):
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
    
create_sized_data(size_initial)
#create_sized_data(500)
#print(len(dict_train_df))

# vocab_processor = tflearn.data_utils.VocabularyProcessor(max_length, min_frequency=0)
#Note : Above function was used instead of the below, which is deprecated. 
# vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_length)

def process_inputs(key, vocab_processor):
    
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

#Converting reviews to ids for all domains and add padding.

# Hyperparameters
min_frequency = 1
max_length = 150

dict_vectorizers = {} #Dict to store the vocab_processor fit on each domain
dict_train_ids = {} #Dict to store train data reviews as sparse matrix of word ids
dict_dev_ids = {} #Dict to store dev data reviews as sparse matrix of word ids
dict_cnn = {} #Dict to store cnn model developed on each domain. Assumes input features are developed using the corresponding count_vectorizer
dict_dev_ypred = {} #Dict to store dev predictions
dict_vocab_len = {} #Store vocab length of each domain
for key in list_df:
    
    #Converting ratings to tokenized word id counts as a sparse matrix using count_vectorizer
    dict_vectorizers[key] = tflearn.data_utils.VocabularyProcessor(max_length, min_frequency=min_frequency)
    print(key)
    dict_train_ids[key], dict_train_y[key],dict_dev_ids[key], dict_dev_ypred[key], dict_vocab_len[key] = process_inputs(key,dict_vectorizers[key])
    
    print("Number words in training corpus for",key,(dict_vocab_len[key]))
    print(key,'dataset id shapes',dict_train_ids[key].shape, dict_dev_ids[key].shape)

    #Print a few examples for viewing
    print('sample review for domain',key, dict_train_df[key].reviewText.iloc[3],'\n')
    print('corresponding ids\n',dict_train_ids[key][3],'\n')

#This code was used to pick max_length for all domains for the CNN, by using a sample of 100000, and a max_length of 10000 for analysis
#it is not needed for running the CNN.
# for key in list_df:
#     length = np.count_nonzero(dict_train_ids[key],axis = 1)
#     print(key,length.shape)
#     print(np.histogram(length,bins = 20))
#     print("Number less than 100",np.count_nonzero(length[length <= 100]))
#     print("Number less than 150",np.count_nonzero(length[length <= 150]))
#     print("Number less than 175",np.count_nonzero(length[length <= 175]))
#     print("Number less than 200",np.count_nonzero(length[length <= 200]))

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
            self.pred_proba = tf.nn.softmax(self.scores, name="pred_proba")
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


        # AUC
#         with tf.name_scope("auc"):
#             false_pos_rate, true_pos_rate, _ = roc_curve(self.input_y, self.pred_proba[:,1])
#             self.auc = auc(false_pos_rate, true_pos_rate)
            
            
        with tf.name_scope('train'):
            #self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate,momentum=momentum,use_nesterov=True).minimize(self.loss)

def batch_generator(ids, labels, batch_size=100, Trainable=False):
            #ids is input, X_train
            #need to fix this to shuffle between epochs
            
            n_batches = len(ids)//batch_size
            ids, labels = ids[:n_batches*batch_size], labels[:n_batches*batch_size]
            if Trainable:
                shuffle = np.random.permutation(np.arange(n_batches*batch_size))
                ids, labels = ids[shuffle], labels[shuffle]
   
            for ii in range(0, len(ids), batch_size):
                yield ids[ii:ii+batch_size], labels[ii:ii+batch_size]

#Model parameters

#embed_dim = 50 #use when not using pre-trained embeddings
embed_dim = hands.shape[1]
filter_sizes= [3,4,5]
num_filters = 256
l2_reg_lambda = 0
learning_rate = 0.01
momentum = 0.9
keep_prob = 0.8
evaluate_train = 2 # of epochs at which to print test accuracy
evaluate_dev = 2 # of epochs at which to estimate and print dev accuracy
time_print = 4 # of epochs at which to print time taken
num_classes = 2
num_epochs = 15
#num_checkpoints = 2
#batch_size = 64
batch_size=128

# out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "cnn"))
# print("Model saving  to {}\n".format(out_dir))

#Actual training loop:

def train_cnn(key, size=5000):
     
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
            size_folder =  "size_" + str(size) 
            out_dir  = os.path.abspath(os.path.join(os.path.curdir, "runs", key, size_folder))
            #out_dir  = os.path.abspath(os.path.join(os.path.curdir, "runs", key))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            model_name = key 
            checkpoint_prefix = os.path.join(checkpoint_dir, model_name  + "_model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())
            
            # Write vocabulary
            ## vocab_processor.save(os.path.join(out_dir, "vocab"))
            
            print('# batches =', len(x_train)//batch_size)
            start = time.time()
            for e in range(num_epochs):
                    
                #sum_scores = np.zeros((batch_size*(len(x_train)//batch_size),1))
                total_loss = 0
                total_acc = 0
                total_auc = 0
                
                for i, (x, y) in enumerate(batch_generator(x_train, y_train, batch_size, Trainable=True), 1):
                    feed = {cnn.input_x: x, cnn.input_y: y, cnn.dropout_keep_prob: keep_prob}
                   # _, loss, accuracy, auc = sess.run([cnn.optimizer,cnn.loss, cnn.accuracy, cnn.auc],feed_dict = feed)
                    _, loss, accuracy = sess.run([cnn.optimizer,cnn.loss, cnn.accuracy],feed_dict = feed)
                    total_loss += loss*len(x)
                    total_acc += accuracy*len(x)
                    
                    #total_auc += auc*len(x)
                    
                if e%evaluate_train==0:
                    avg_loss = total_loss/(batch_size*(len(x_train)//batch_size))
                    avg_acc = total_acc/(batch_size*(len(x_train)//batch_size))
                    #avg_auc = total_auc/(batch_size*(len(x_train)//batch_size))
                   # print("Train epoch {}, average loss {:g}, average accuracy {:g},average auc {:g}".format(e, avg_loss, avg_acc, avg_auc))
                    print("Train epoch {}, average loss {:g}, average accuracy {:g},".format(e, avg_loss, avg_acc))

                if e%evaluate_dev==0:
                    
                    total_loss = 0
                    total_acc = 0
                    num_batches = 0
                    total_auc = 0
                    y_pred = []
                    y_pred_proba = []
                    y_shuffled = []
                    total_batch_acc = 0
                    
                    for ii, (x, y) in enumerate(batch_generator(x_dev, y_dev, batch_size, Trainable=False), 1):
                        
                        feed_dict = {cnn.input_x: x, cnn.input_y: y, cnn.dropout_keep_prob: 1.0}
                        #loss, accuracy, auc = sess.run([cnn.loss, cnn.accuracy, cnn.auc],feed_dict)
                       # batch_pred,batch_pred_proba,loss, accuracy  = sess.run([cnn.loss, cnn.accuracy],feed_dict)
                        batch_pred,batch_pred_proba,loss, accuracy  = sess.run([cnn.predictions, cnn.pred_proba, cnn.loss, cnn.accuracy],feed_dict)
                        total_loss += loss*len(x)
                        total_acc += accuracy*len(x)
                        
                        batch_accuracy= np.sum(y==batch_pred)/y.shape[0]
                        total_batch_acc += batch_accuracy
                        y_pred= np.concatenate([y_pred, batch_pred])
                        y_pred_proba= np.concatenate([y_pred_proba, batch_pred_proba[:,1]])
                        y_shuffled = np.concatenate([y_shuffled, y])
                        
                        num_batches += 1
                        
                    avg_loss = total_loss/(num_batches*batch_size)
                    avg_acc = total_acc/(num_batches*batch_size)
                    
                    print('y_dev.shape',y_dev.shape)
                    print('y_shuffled.shape',y_shuffled.shape)
                    
                    if np.array_equal(y_shuffled,y_dev):
                        print("Yes")
                    right_acc = total_batch_acc/(num_batches)
                    #avg_auc = total_auc/(num_batches*batch_size)
                    
                    #Calculate Accuracy
                    new_acc = accuracy_score(y_shuffled, y_pred, normalize=True ) 
                     
                    
                    false_pos_rate, true_pos_rate, _ = roc_curve(y_shuffled, y_pred_proba)  
                    roc_auc = auc(false_pos_rate, true_pos_rate)
                    
                #time_str = datetime.datetime.now().isoformat()
                    print("\t\tDev epoch {}, average loss {:g}, average accuracy {:g},".format(e, avg_loss, avg_acc))
                    print("\t\tDev epoch {}, auc {:g}, new accuracy {:g}, right accuracy {:g},".format(e,  roc_auc, new_acc, right_acc))
                    #print("\t\tDev epoch {}, average loss {:g}, average accuracy {:g},average auc {:g}".format(e, avg_loss, avg_acc, avg_auc))
                if e%time_print == 0:
                    end = time.time()
                    print("\t\t\t\t    Time taken for",e,"epochs = ", end-start)
                    
                    
        # Save model weights for future use.
        
        
            #save_path = saver.save(sess, checkpoint_prefix, global_step=20,write_meta_graph=False)
            save_path = saver.save(sess, checkpoint_prefix)
            print("Saved model", model_name, save_path)
            
            #calculate predictions and prediction probability    
#             feed_dict={cnn.input_x:x_dev, cnn.input_y: y_dev, cnn.dropout_keep_prob: 1.0}
#             y_pred, y_pred_proba = sess.run([cnn.predictions, cnn.pred_proba],feed_dict)
            #print(y_pred, y_pred_proba)

list_df = ['toys','vid','aut','hnk'] 

#Create and train the cnn models for all 4 domains
#Pass the size to save the model name with size in different folders

size_train = size_initial
for key in list_df:
    print(key, size_train)
    train_cnn(key, size=size_train)
    

def predict_accuracy(src_key, size, tar_key):
    
    batch_size=50
    print('target',tar_key,'source', src_key)
    V = dict_vocab_len[tar_key]
    
    size_folder =  "size_" + str(size) 
    out_dir  = os.path.abspath(os.path.join(os.path.curdir, "runs", src_key, size_folder))
    #out_dir  = os.path.abspath(os.path.join(os.path.curdir, "runs", src_key))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

    
    print(checkpoint_dir)
    src_model = src_key
    #graph_meta_file = checkpoint_dir + '/' + 'hnk01_model.meta'
    graph_meta_file = checkpoint_dir + '/' + src_model +'_model.meta'
    graph=tf.Graph()

    with graph.as_default():
        with tf.Session() as sess:
    
      #new_saver = tf.train.import_meta_graph(checkpoint_dir/'hnk_model.meta')
            new_saver = tf.train.import_meta_graph(graph_meta_file)
            new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    
            x_dev = dict_dev_ids[tar_key]      
            y_dev = dict_dev_ypred[tar_key]
        
            #create graph from saved model
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        
            pred_proba = graph.get_operation_by_name("output/pred_proba").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        
            y_pred = []
            y_pred_proba = []
            total_batch_acc = 0
            num_batches = 0
            y_shuffled = []
            abs_y_pred_proba = []
            for ii, (x, y) in enumerate(batch_generator(x_dev, y_dev, batch_size, Trainable=False), 1):
                        
                feed_dict = {input_x: x, input_y: y, dropout_keep_prob: 1.0}
                batch_pred, batch_pred_proba  = sess.run([ predictions, pred_proba],feed_dict)
                batch_accuracy= np.sum(y==batch_pred)/y.shape[0]
                total_batch_acc += batch_accuracy
                y_pred= np.concatenate([y_pred, batch_pred])
                y_pred_proba= np.concatenate([y_pred_proba, batch_pred_proba[:,1]])
                abs_y_pred_proba = np.concatenate([abs_y_pred_proba,np.absolute(batch_pred_proba[:,1] - batch_pred_proba[:,0])])
                y_shuffled = np.concatenate([y_shuffled, y])

                num_batches += 1           
        
            # Calculate auc
            # false_pos_rate, true_pos_rate, _ = roc_curve(y_dev, y_pred_proba[:,1])
            false_pos_rate, true_pos_rate, _ = roc_curve(y_shuffled, y_pred_proba)  
            roc_auc = auc(false_pos_rate, true_pos_rate)
            # print(src_key, tar_key, "AUC","{:.02%}".format(roc_auc))
            
            #Calculate Accuracy
            acc = accuracy_score(y_shuffled, y_pred, normalize=True )
            #print('source',src_key, 'target',tar_key, "accuracy","{:.02%}".format(acc))
            #print("")
            f1_pos = f1_score(y_shuffled, y_pred, average = None)[1]
            f1_neg = f1_score(y_shuffled, y_pred, average = None)[0]
            f1_avg = f1_score(y_shuffled, y_pred, average = 'macro')
        
        #Save absolute_y_pred_proba
        
        #check if the batching process left remainders. This will result in incorrect length of y_pred_proba saved
        if y_dev.shape[0] != abs_y_pred_proba.shape[0]:
            print("Length of y_pred_proba does not match y_dev. Fix batch_size")
            print("Pred proba file not saved")
#         else:    
#             file_name = "src_" + src_key + "_tar_" + tar_key + "_" + str(y_dev.shape[0])
#             np.savez_compressed('test_file',pred_prob=abs_y_pred_proba)
#             print( file_name, "Saved file successfully")
    return acc, roc_auc, f1_pos, f1_neg, f1_avg

#calculate transfer accuracy for all domains, and save results in a dataframe
list_df = ['toys','vid','aut','hnk'] 
#size = size_train
size = size_initial
transfer_results = pd.DataFrame(index=list_df,columns=list_df) #Dataframe to store accuracy on transfer. Col = Model, row = dataframe
transfer_results_auc = pd.DataFrame(index=list_df,columns=list_df) #Dataframe to store AUC on transfer. Col = Model, row = dataframe
transfer_results_f1_pos = pd.DataFrame(index=list_df,columns=list_df) #Dataframe to store f1-positive on transfer.
transfer_results_f1_neg = pd.DataFrame(index=list_df,columns=list_df) #Dataframe to store f1-negative on transfer.
transfer_results_f1_avg = pd.DataFrame(index=list_df,columns=list_df) #Dataframe to store f1-average on transfer.

for s_key in list_df:
    for t_key in list_df:
        acc, roc_auc, f1_pos, f1_neg, f1_avg = predict_accuracy(s_key,size, t_key)
        transfer_results[s_key][t_key] = acc
        transfer_results_auc[s_key][t_key] = roc_auc
        transfer_results_f1_pos[s_key][t_key] = f1_pos
        transfer_results_f1_neg[s_key][t_key] = f1_neg
        transfer_results_f1_avg[s_key][t_key] = f1_avg

print('\n Accuracy with transfer\n',transfer_results)
print('\n AUC with transfer\n',transfer_results_auc)
print('\n F1-positive with transfer\n',transfer_results_f1_pos)
print('\n F1-negative with transfer\n',transfer_results_f1_neg)
print('\n F1-average with transfer\n',transfer_results_f1_avg)

dict_transfer_vect = {} #Dictionary to store two domain vocab_vectorizer
dict_transfer_train_ids = {} #Dictionary to store review ids of train set based on on two domains vocab_vectorizer
dict_transfer_dev_ids = {} #Dictionary to store review ids of train set based on on two domains vocab_vectorizer
for s_key in list_df:
    dict_transfer_vect[s_key] = {}
    dict_transfer_train_ids[s_key] = {}
    dict_transfer_dev_ids[s_key] = {}

#Note : size of src and tgt needs to be less than the original size read into dict_train_df with create_sized data.

def process_transfer_data(src_key,tgt_key, size_train = 10000):
      
    #Create combined dataframe of reviewText from both domains
    tmp_src_df = dict_train_df[src_key][:size_train] #picking the right sized subset from dict_train_df, dict_train_y
    tmp_tgt_df = dict_train_df[tgt_key][:size_train]
    tmp_src_df_dev = dict_dev_df[src_key][:np.int(size_train*0.3)] #picking the right sized subset from dict_train_df, dict_train_y
    tmp_tgt_df_dev = dict_dev_df[tgt_key][:np.int(size_train*0.3)]
    #print(tmp_src_df.shape,tmp_tgt_df.shape,tmp_src_df_dev.shape,tmp_tgt_df_dev.shape)
    temp_two_df_reviews = pd.concat([tmp_src_df.reviewText,tmp_tgt_df.reviewText])
    #print('combined df shape for',src_key,tgt_key,temp_two_df_reviews.shape)
                
    #create countVectorizer on combined dataframe of reviewText from both domains
    dict_transfer_vect[src_key][tgt_key] = tflearn.data_utils.VocabularyProcessor(max_length, min_frequency=min_frequency)
    dict_transfer_vect[src_key][tgt_key] = dict_transfer_vect[src_key][tgt_key].fit(temp_two_df_reviews)
    print("Number words in training corpus for keys",src_key,tgt_key,len(dict_transfer_vect[src_key][tgt_key].vocabulary_))
                
    #create id vectors of reviews for each df, train and dev set, using combined countVectorizer
    #create id vectors of reviews for each df, train and dev set, using combined countVectorizer
    dict_transfer_train_ids[src_key][tgt_key] = dict_transfer_vect[src_key][tgt_key].transform(tmp_src_df.reviewText)
    dict_transfer_train_ids[tgt_key][src_key] = dict_transfer_vect[src_key][tgt_key].transform(tmp_tgt_df.reviewText)
    dict_transfer_dev_ids[src_key][tgt_key] = dict_transfer_vect[src_key][tgt_key].transform(tmp_src_df_dev.reviewText)
    dict_transfer_dev_ids[tgt_key][src_key] = dict_transfer_vect[src_key][tgt_key].transform(tmp_tgt_df_dev.reviewText)
    # x_train = np.array(list(x_train))
    dict_transfer_train_ids[src_key][tgt_key] = np.array(list(dict_transfer_train_ids[src_key][tgt_key]))
    dict_transfer_train_ids[tgt_key][src_key] = np.array(list(dict_transfer_train_ids[tgt_key][src_key]))
    dict_transfer_dev_ids[src_key][tgt_key] = np.array(list(dict_transfer_dev_ids[src_key][tgt_key]))
    dict_transfer_dev_ids[tgt_key][src_key] = np.array(list(dict_transfer_dev_ids[tgt_key][src_key]))

#Convert the train and dev data to ids using the combined source and target domain vocab_vectorizer
size_initial = size_initial
list_src = ['vid']
list_tgt = ['aut']
for s_key in list_src:
    #print(s_key)
    for t_key in list_tgt:
        print('source key',s_key, 'target key',t_key)
        process_transfer_data(s_key,t_key, size_train = size_initial)
        print(s_key,'train set shape',dict_transfer_train_ids[s_key][t_key].shape)
        print(t_key,'train set shape',dict_transfer_train_ids[t_key][s_key].shape)
        print(s_key,'dev set shape',dict_transfer_dev_ids[s_key][t_key].shape)
        print(t_key,'dev set shape',dict_transfer_dev_ids[t_key][s_key].shape)
#         print(dict_train_df[s_key]['reviewText'].iloc[1])
#         print(dict_transfer_train_ids[s_key][t_key][1])
#         print(dict_train_df[t_key]['reviewText'].iloc[1])
#         print(dict_transfer_train_ids[t_key][s_key][1])
        
        

def train_transfer_cnn(skey,tkey,size=10000):
     
    x_train =  dict_transfer_train_ids[skey][tkey][:size]
    y_train = dict_train_y[skey][:size]
    x_dev = dict_transfer_dev_ids[skey][tkey][:np.int(size*0.3)] #Note : 0.3 is hard coded as the relative size of dev vs train.
    y_dev = dict_dev_ypred[skey][:np.int(size*0.3)]
    x_dev_tgt = dict_transfer_dev_ids[tkey][skey][:np.int(size*0.3)]
    y_dev_tgt = dict_dev_ypred[tkey][:np.int(size*0.3)]
    V = len(dict_transfer_vect[skey][tkey].vocabulary_)
    
    with tf.Graph().as_default():

        with tf.Session() as sess:
        
            cnn = TextCNN(sequence_length=x_train.shape[1], num_classes=num_classes, vocab_size=V, learning_rate = learning_rate,
                        momentum = momentum, embedding_size=embed_dim, gl_embed = hands.W, filter_sizes= filter_sizes, 
                      num_filters=num_filters, l2_reg_lambda=l2_reg_lambda)
            
            sess.run(tf.global_variables_initializer())
            print('completed cnn creation')

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            size_folder =  "size_" + str(size) 
            out_dir  = os.path.abspath(os.path.join(os.path.curdir, "runs", skey, tkey, size_folder))
            #out_dir  = os.path.abspath(os.path.join(os.path.curdir, "runs", key))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            model_name = ''.join([skey, tkey])
            checkpoint_prefix = os.path.join(checkpoint_dir, model_name  + "_model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())
            
            # Write vocabulary
            ## vocab_processor.save(os.path.join(out_dir, "vocab"))
            
            tmp_ix = [2*e for e in range(np.int((num_epochs+2)/2))]
            results = pd.DataFrame(index = tmp_ix,columns = ['size','acc','f1_avg','auc','f1_pos','f1_neg'])
            print('# batches =', len(x_train)//batch_size)
            start = time.time()
            for e in range(num_epochs):
                    
                #sum_scores = np.zeros((batch_size*(len(x_train)//batch_size),1))
                total_loss = 0
                total_acc = 0
                total_auc = 0
                
                for i, (x, y) in enumerate(batch_generator(x_train, y_train, batch_size, Trainable=True), 1):
                    feed = {cnn.input_x: x, cnn.input_y: y, cnn.dropout_keep_prob: keep_prob}
                   # _, loss, accuracy, auc = sess.run([cnn.optimizer,cnn.loss, cnn.accuracy, cnn.auc],feed_dict = feed)
                    _, loss, accuracy = sess.run([cnn.optimizer,cnn.loss, cnn.accuracy],feed_dict = feed)
                    total_loss += loss*len(x)
                    total_acc += accuracy*len(x)
                    
                    #total_auc += auc*len(x)
                    
                if e%evaluate_train==0:
                    avg_loss = total_loss/(batch_size*(len(x_train)//batch_size))
                    avg_acc = total_acc/(batch_size*(len(x_train)//batch_size))
                    print("Train epoch {}, average loss {:g}, average accuracy {:g},".format(e, avg_loss, avg_acc))

                if e%evaluate_dev==0:
                    
                    total_loss = 0
                    total_acc = 0
                    num_batches = 0
                    total_auc = 0
                    y_pred = []
                    y_pred_proba = []
                    y_shuffled = []
                    total_batch_acc = 0
                    
                    for ii, (x, y) in enumerate(batch_generator(x_dev, y_dev, batch_size, Trainable=False), 1):
                        
                        feed_dict = {cnn.input_x: x, cnn.input_y: y, cnn.dropout_keep_prob: 1.0}
                        #loss, accuracy, auc = sess.run([cnn.loss, cnn.accuracy, cnn.auc],feed_dict)
                       # batch_pred,batch_pred_proba,loss, accuracy  = sess.run([cnn.loss, cnn.accuracy],feed_dict)
                        batch_pred,batch_pred_proba,loss, accuracy  = sess.run([cnn.predictions, cnn.pred_proba, cnn.loss, cnn.accuracy],feed_dict)
                        total_loss += loss*len(x)
                        total_acc += accuracy*len(x)
                        
                        batch_accuracy= np.sum(y==batch_pred)/y.shape[0]
                        total_batch_acc += batch_accuracy
                        y_pred= np.concatenate([y_pred, batch_pred])
                        y_pred_proba= np.concatenate([y_pred_proba, batch_pred_proba[:,1]])
                        y_shuffled = np.concatenate([y_shuffled, y])
                        
                        num_batches += 1
                        
                    avg_loss = total_loss/(num_batches*batch_size)
                    avg_acc = total_acc/(num_batches*batch_size)
                    
#                     print('y_dev.shape',y_dev.shape)
#                     print('y_shuffled.shape',y_shuffled.shape)
                    
                    if np.array_equal(y_shuffled,y_dev):
                        print("Yes")
                    #right_acc = total_batch_acc/(num_batches)
                    #avg_auc = total_auc/(num_batches*batch_size)
                    
                    #Calculate Accuracy
                    #new_acc = accuracy_score(y_shuffled, y_pred, normalize=True )                   
                    false_pos_rate, true_pos_rate, _ = roc_curve(y_shuffled, y_pred_proba)  
                    roc_auc = auc(false_pos_rate, true_pos_rate)
                    f1_pos = f1_score(y_shuffled, y_pred, average = None)[1]
                    f1_neg = f1_score(y_shuffled, y_pred, average = None)[0]
                    f1_avg = f1_score(y_shuffled, y_pred, average = 'macro')
                    
                    results['acc'][e] = avg_acc
                    results['f1_avg'][e] = f1_avg
                    results['auc'][e] = roc_auc
                    results['f1_pos'][e] = f1_pos
                    results['f1_neg'][e] =  f1_neg
                
                    
                #time_str = datetime.datetime.now().isoformat()
                    print("\tDev epoch %d,average loss %0.3f,average accuracy %0.3f,auc %0.3f,f1_pos %0.3f,f1_neg %0.3f,f1_avg %0.3f"
                          %(e, avg_loss, avg_acc, roc_auc,f1_pos,f1_neg,f1_avg))
                if e%time_print == 0:
                    end = time.time()
                    print("\t\t\t\t    Time taken for",e,"epochs = ", end-start)
                    
                    
            #Estimate accuracy on target dev set
            total_loss = 0
            total_acc = 0
            num_batches = 0
            total_auc = 0
            y_pred = []
            y_pred_proba = []
            y_shuffled = []
            total_batch_acc = 0
            
            
            #Anamika added code for error analysis
            y_pred_proba_pos = []
            y_pred_proba_neg = []
            
            
            for ii, (x, y) in enumerate(batch_generator(x_dev_tgt, y_dev_tgt, batch_size, Trainable=False), 1):

                feed_dict = {cnn.input_x: x, cnn.input_y: y, cnn.dropout_keep_prob: 1.0}
                #loss, accuracy, auc = sess.run([cnn.loss, cnn.accuracy, cnn.auc],feed_dict)
                # batch_pred,batch_pred_proba,loss, accuracy  = sess.run([cnn.loss, cnn.accuracy],feed_dict)
                batch_pred,batch_pred_proba,loss, accuracy  = sess.run([cnn.predictions, cnn.pred_proba, cnn.loss, cnn.accuracy],feed_dict)
                total_loss += loss*len(x)
                total_acc += accuracy*len(x)

                batch_accuracy= np.sum(y==batch_pred)/y.shape[0]
                total_batch_acc += batch_accuracy
                y_pred= np.concatenate([y_pred, batch_pred])
                y_pred_proba= np.concatenate([y_pred_proba, batch_pred_proba[:,1]])
                y_shuffled = np.concatenate([y_shuffled, y])
                y_pred_proba_pos= np.concatenate([y_pred_proba_pos, batch_pred_proba[:,1]])
                y_pred_proba_neg= np.concatenate([y_pred_proba_neg, batch_pred_proba[:,0]])
                
                num_batches += 1
                    
            avg_loss = total_loss/(num_batches*batch_size)
            avg_acc = total_acc/(num_batches*batch_size)

#             print('y_dev.shape',y_dev.shape)
#             print('y_shuffled.shape',y_shuffled.shape)

            if np.array_equal(y_shuffled,y_dev):
                print("Yes")
                #right_acc = total_batch_acc/(num_batches)

            #Calculate Accuracy, AUC
            #new_acc = accuracy_score(y_shuffled, y_pred, normalize=True ) 
            false_pos_rate, true_pos_rate, _ = roc_curve(y_shuffled, y_pred_proba)  
            roc_auc = auc(false_pos_rate, true_pos_rate)
            f1_pos = f1_score(y_shuffled, y_pred, average = None)[1]
            f1_neg = f1_score(y_shuffled, y_pred, average = None)[0]
            f1_avg = f1_score(y_shuffled, y_pred, average = 'macro')
            #print("\t\t",tkey,"Dev epoch {}, average loss {:g}, average accuracy {:g},".format(e, avg_loss, avg_acc))
            print("\t\t",tkey,"Dev epoch %d, average loss %0.3f,average accuracy %0.3f,auc %0.3f,f1_pos %0.3f,f1_neg %0.3f,f1_avg %0.3f"
                  %(e, avg_loss, avg_acc, roc_auc,f1_pos,f1_neg,f1_avg))

        # Save model weights for future use.       
            #save_path = saver.save(sess, checkpoint_prefix, global_step=20,write_meta_graph=False)
            save_path = saver.save(sess, checkpoint_prefix)
            print("Saved model", model_name, save_path)
            
            
            
            
            ##Anamika added code 
            
              #for error analysis
        print("ERROR ANALYSIS")
        print(len(y_shuffled))
        src_key = skey
        tar_key = tkey
        print("src_key", skey, "tar_key", tkey)
        pos_err_pos = np.where((y_shuffled != y_pred) & (y_shuffled ==1))
        neg_err_pos = np.where((y_shuffled != y_pred) & (y_shuffled ==0))
        
        
        
        #for actual negatives that model predicted negatives(true negatives)
        print("True negatives")
        #no_err_neg_probas = y_pred_proba_neg[np.where((y_shuffled == y_pred) & (y_shuffled ==0))]
        no_err_neg_positions = np.where((y_shuffled == y_pred) & (y_shuffled ==0))
        no_err_neg_probas = y_pred_proba_neg[no_err_neg_positions]
        print("Values in no_err_neg_probas", len(no_err_neg_probas))
        #neg_err_pos_vals = 
        #if y_shuffled ==0 and y_pred == 1:
        print("Correct neg probabilities > 0.9", len(no_err_neg_probas[no_err_neg_probas >= 0.9]))
        pos_nine_buckets = np.where((y_shuffled == y_pred) & (y_shuffled ==0) & (y_pred_proba_neg >= 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_nine_buckets[0][n]], "Pred y", y_pred[pos_nine_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_nine_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_nine_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_nine_buckets[0][n]]))
            
            print(dict_dev_df[t_key].reviewText.iloc[pos_nine_buckets[0][n]],'\n')
        
        print("Correct neg probabilities between 0.8 and 0.9", len(no_err_neg_probas[no_err_neg_probas >= 0.8 ])- len(no_err_neg_probas[no_err_neg_probas >= 0.9]))
        pos_eight_buckets = np.where((y_shuffled == y_pred) & (y_shuffled ==0) & (y_pred_proba_neg >= 0.8) & (y_pred_proba_neg < 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_eight_buckets[0][n]], "Pred y", y_pred[pos_eight_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_eight_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_eight_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_eight_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_eight_buckets[0][n]],'\n')
        print("Correct neg probabilities between 0.7 and 0.8", len(no_err_neg_probas[no_err_neg_probas >= 0.7 ])- len(no_err_neg_probas[no_err_neg_probas >= 0.8]))
        pos_seven_buckets = np.where((y_shuffled == y_pred) & (y_shuffled ==0) & (y_pred_proba_neg >= 0.7) & (y_pred_proba_neg < 0.8))
        
        for n in range(2):
            print("actual y", y_shuffled[pos_seven_buckets[0][n]], "Pred y", y_pred[pos_seven_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_seven_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_seven_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_seven_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_eight_buckets[0][n]],'\n')
        print("Correct neg probabilities between 0.6 and 0.7", len(no_err_neg_probas[no_err_neg_probas >= 0.6 ])- len(no_err_neg_probas[no_err_neg_probas >= 0.7]))
        print("Correct neg probabilities between 0.5 and 0.6", len(no_err_neg_probas[no_err_neg_probas >= 0.5 ])- len(no_err_neg_probas[no_err_neg_probas >= 0.6]))
        print("Correct neg probabilities < 0.5", len(no_err_neg_probas[no_err_neg_probas <0.5 ]))
        
        print("")
        print("")
        
        #for actual negatives that model predicted positives(False positives)
        print("False positives")
        neg_err_pos_probas = y_pred_proba_pos[neg_err_pos]
        print("Values in neg_err_pos_probas", len(neg_err_pos_probas))
        #neg_err_pos_vals = 
        #if y_shuffled ==0 and y_pred == 1:
        print("Pos probabilities > 0.9", len(neg_err_pos_probas[neg_err_pos_probas >= 0.9]))
        pos_nine_buckets = np.where((y_shuffled != y_pred) & (y_shuffled ==0) & (y_pred_proba_pos >= 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_nine_buckets[0][n]], "Pred y", y_pred[pos_nine_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_nine_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_nine_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_nine_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_nine_buckets[0][n]],'\n')
        
        print("Pos probabilities between 0.8 and 0.9", len(neg_err_pos_probas[neg_err_pos_probas >= 0.8 ])- len(neg_err_pos_probas[neg_err_pos_probas >= 0.9]))
        pos_eight_buckets = np.where((y_shuffled != y_pred) & (y_shuffled ==0) & (y_pred_proba_pos >= 0.8) & (y_pred_proba_pos < 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_eight_buckets[0][n]], "Pred y", y_pred[pos_eight_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_eight_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_eight_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_eight_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_eight_buckets[0][n]],'\n')
            
        print("Pos probabilities between 0.7 and 0.8", len(neg_err_pos_probas[neg_err_pos_probas >= 0.7 ])- len(neg_err_pos_probas[neg_err_pos_probas >= 0.8]))
        pos_seven_buckets = np.where((y_shuffled != y_pred) & (y_shuffled ==0) & (y_pred_proba_pos >= 0.7) & (y_pred_proba_pos < 0.8))
        for n in range(2):
            print("actual y", y_shuffled[pos_seven_buckets[0][n]], "Pred y", y_pred[pos_seven_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_seven_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_seven_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_seven_buckets[0][n]]))
            print(dict_dev_df[tar_key].reviewText.iloc[pos_seven_buckets[0][n]],'\n')
        print("Pos probabilities between 0.6 and 0.7", len(neg_err_pos_probas[neg_err_pos_probas >= 0.6 ])- len(neg_err_pos_probas[neg_err_pos_probas >= 0.7]))
        print("Pos probabilities between 0.5 and 0.6", len(neg_err_pos_probas[neg_err_pos_probas >= 0.5 ])- len(neg_err_pos_probas[neg_err_pos_probas >= 0.6]))
        print("Pos probabilities < 0.5", len(neg_err_pos_probas[neg_err_pos_probas <0.5 ]))
        
        print("")
        print("")
        
        
        
        
  #for actual positives that model predicted positives(true positives)
        print("True positives")
        #no_err_neg_probas = y_pred_proba_neg[np.where((y_shuffled == y_pred) & (y_shuffled ==0))]
        no_err_pos_positions = np.where((y_shuffled == y_pred) & (y_shuffled ==1))
        no_err_pos_probas = y_pred_proba_pos[no_err_pos_positions]
        print("Values in no_err_pos_probas", len(no_err_pos_probas))
        #neg_err_pos_vals = 
        #if y_shuffled ==0 and y_pred == 1:
        print("Correct positivr probabilities > 0.9", len(no_err_pos_probas[no_err_pos_probas >= 0.9]))
        pos_nine_buckets = np.where((y_shuffled == y_pred) & (y_shuffled ==1) & (y_pred_proba_pos >= 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_nine_buckets[0][n]], "Pred y", y_pred[pos_nine_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_nine_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_nine_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_nine_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_nine_buckets[0][n]],'\n')
        
        print("Correct pos probabilities between 0.8 and 0.9", len(no_err_pos_probas[no_err_pos_probas >= 0.8 ])- len(no_err_pos_probas[no_err_pos_probas >= 0.9]))
        pos_eight_buckets = np.where((y_shuffled == y_pred) & (y_shuffled ==1) & (y_pred_proba_pos >= 0.8) & (y_pred_proba_pos < 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_eight_buckets[0][n]], "Pred y", y_pred[pos_eight_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_eight_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_eight_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_eight_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_eight_buckets[0][n]],'\n')
        print("Correct pos probabilities between 0.7 and 0.8", len(no_err_pos_probas[no_err_pos_probas >= 0.7 ])- len(no_err_pos_probas[no_err_pos_probas >= 0.8]))
        pos_seven_buckets = np.where((y_shuffled == y_pred) & (y_shuffled ==1) & (y_pred_proba_pos >= 0.7) & (y_pred_proba_pos < 0.8))
        for n in range(2):
            print("actual y", y_shuffled[pos_seven_buckets[0][n]], "Pred y", y_pred[pos_seven_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_seven_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_seven_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_seven_buckets[0][n]]))
            print(dict_dev_df[tar_key].reviewText.iloc[pos_seven_buckets[0][n]],'\n')
        print("Correct pos probabilities between 0.6 and 0.7", len(no_err_pos_probas[no_err_pos_probas >= 0.6 ])- len(no_err_pos_probas[no_err_pos_probas >= 0.7]))
        print("Correct pos probabilities between 0.5 and 0.6", len(no_err_pos_probas[no_err_pos_probas >= 0.5 ])- len(no_err_pos_probas[no_err_pos_probas >= 0.6]))
        print("Correct pos probabilities < 0.5", len(no_err_pos_probas[no_err_pos_probas <0.5 ]))
        
        print("")
        print("")    
        
        
#for actual positives that model predicted as negatives(False negatives)y_pred is 0

        
        print("False negatives")
        pos_err_neg_probas = y_pred_proba_neg[pos_err_pos]
        print("Values in pos_err_neg_probas", len(pos_err_neg_probas))
        #neg_err_pos_vals = 
        #if y_shuffled ==0 and y_pred == 1:
        print("Neg probabilities > 0.9", len(pos_err_neg_probas[pos_err_neg_probas >= 0.9]))
        pos_nine_buckets = np.where((y_shuffled != y_pred) & (y_shuffled ==1) & (y_pred_proba_neg >= 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_nine_buckets[0][n]], "Pred y", y_pred[pos_nine_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_nine_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_nine_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_nine_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_nine_buckets[0][n]],'\n')
        
        print("Neg probabilities between 0.8 and 0.9", len(pos_err_neg_probas[pos_err_neg_probas >= 0.8 ])- len(pos_err_neg_probas[pos_err_neg_probas >= 0.9]))
        pos_eight_buckets = np.where((y_shuffled != y_pred) & (y_shuffled ==1) & (y_pred_proba_neg > 0.8) & (y_pred_proba_neg < 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_eight_buckets[0][n]], "Pred y", y_pred[pos_eight_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_eight_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_eight_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_eight_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_eight_buckets[0][n]],'\n')
        print("Pos probabilities between 0.7 and 0.8", len(pos_err_neg_probas[pos_err_neg_probas >= 0.7 ])- len(pos_err_neg_probas[pos_err_neg_probas >= 0.8]))
        pos_seven_buckets = np.where((y_shuffled != y_pred) & (y_shuffled ==1) & (y_pred_proba_neg > 0.7) & (y_pred_proba_neg < 0.8))
        for n in range(2):
            print("actual y", y_shuffled[pos_seven_buckets[0][n]], "Pred y", y_pred[pos_eight_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_seven_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_seven_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_seven_buckets[0][n]]))
            print(dict_dev_df[tar_key].reviewText.iloc[pos_seven_buckets[0][n]],'\n')
        print("Pos probabilities between 0.6 and 0.7", len(pos_err_neg_probas[pos_err_neg_probas >= 0.6 ])- len(pos_err_neg_probas[pos_err_neg_probas >= 0.7]))
        print("Pos probabilities between 0.5 and 0.6", len(pos_err_neg_probas[pos_err_neg_probas >= 0.5 ])- len(pos_err_neg_probas[pos_err_neg_probas >= 0.6]))
        print("Pos probabilities < 0.5", len(pos_err_neg_probas[pos_err_neg_probas <0.5 ]))
        
    return results
            
 

#num_epochs = 1
num_epochs = 15
results_transfer = pd.DataFrame()
s_key = 'vid' #Note : these need to be the same or a subset of the keys in the process_transfer input function which does the combined vocabulary preprocessing.
t_key = 'aut'
print('source key',s_key, 'target key',t_key)
results = train_transfer_cnn(s_key,t_key,size_initial)
results['size'] = size_initial
results_transfer = pd.concat([results_transfer,results])
results_transfer

#Updated continue_train for adding samples from target domain to continue to train on source domain.
def continue_transfer_train(skey,size,tkey,tgt_train_df,tgt_train_y): 
#Note size is size of source domain train set for picking the right sized model parameters
    
    #out_dir  = os.path.abspath(os.path.join(os.path.curdir, "runs", src_key))
    #out_dir  = os.path.abspath(os.path.join(os.path.curdir, "testruns", src_key))
    
    size_folder =  "size_" + str(size) 
    out_dir  = os.path.abspath(os.path.join(os.path.curdir, "runs", skey, tkey, size_folder))
    #saved model being picked is the one that was trained on source domain, but with the vocabulary of both domains combined
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    print(checkpoint_dir)
    src_model = ''.join([skey, tkey])
    #graph_meta_file = checkpoint_dir + '/' + 'hnk01_model.meta'
#     graph_meta_file = checkpoint_dir + '/' + src_model +'01_model.meta'
    graph=tf.Graph()
    
    x_train = tgt_train_df
    y_train = tgt_train_y
    x_dev = dict_transfer_dev_ids[tkey][skey][:np.int(size*0.3)]
    y_dev = dict_dev_ypred[tkey][:np.int(size*0.3)]
    V = len(dict_transfer_vect[skey][tkey].vocabulary_)
    
    #create a dataframe to store the results together and pass back out of the function
    tmp_ix = [2*e for e in range(int((num_epochs+2)/2))]
    results = pd.DataFrame(index = tmp_ix,columns = ['size','acc','auc','f1_neg','f1_pos','f1_avg'])
       
    with graph.as_default():
        with tf.Session() as sess:           
            cnn = TextCNN(sequence_length=x_train.shape[1], num_classes=num_classes, vocab_size=V, learning_rate = learning_rate,
                        momentum = momentum, embedding_size=embed_dim, gl_embed = hands.W, filter_sizes= filter_sizes, 
                      num_filters=num_filters, l2_reg_lambda=l2_reg_lambda)
            
            sess.run(tf.global_variables_initializer())
 
            saver = tf.train.Saver()
    
          #new_saver = tf.train.import_meta_graph(checkpoint_dir/'hnk_model.meta')
#             new_saver = tf.train.import_meta_graph(graph_meta_file)
#             new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            
            
            #initializing weights from a previous session 
            initialising_model = src_model+'_model'
            print(" RESTORING SESSION FOR WEIGHTS INITIALIZATION")
            # Exclude output layer weights from variables we will restore
            variables_to_restore = [v for v in tf.global_variables()]
            # Replace variables scope with that of the current model
            loader = tf.train.Saver({v.op.name.replace(src_model, initialising_model): v for v in variables_to_restore})
            load_path = checkpoint_dir + '/' + initialising_model 
            #load_path = checkpoint_dir  
            loader.restore(sess, load_path)
            print(" Model loaded from: " + load_path) 
            print('# batches =', len(x_train)//batch_size)
            start = time.time()
           
            for e in range(num_epochs):
                    
                #sum_scores = np.zeros((batch_size*(len(x_train)//batch_size),1))
                total_loss = 0
                total_acc = 0
                total_auc = 0
                for i, (x, y) in enumerate(batch_generator(x_train, y_train, batch_size, Trainable=True), 1):
                    feed = {cnn.input_x: x, cnn.input_y: y, cnn.dropout_keep_prob: keep_prob}
                   # _, loss, accuracy, auc = sess.run([cnn.optimizer,cnn.loss, cnn.accuracy, cnn.auc],feed_dict = feed)
                    _, loss, accuracy = sess.run([cnn.optimizer,cnn.loss, cnn.accuracy],feed_dict = feed)
                    total_loss += loss*len(x)
                    total_acc += accuracy*len(x)
                    
                    #total_auc += auc*len(x)
                    
                if e%evaluate_train==0:
                    avg_loss = total_loss/(batch_size*(len(x_train)//batch_size))
                    avg_acc = total_acc/(batch_size*(len(x_train)//batch_size))
                    #avg_auc = total_auc/(batch_size*(len(x_train)//batch_size))
                   # print("Train epoch {}, average loss {:g}, average accuracy {:g},average auc {:g}".format(e, avg_loss, avg_acc, avg_auc))
                    print("Train epoch {}, average loss {:g}, average accuracy {:g},".format(e, avg_loss, avg_acc))

                if e%evaluate_dev==0:
                    
                    total_loss = 0
                    total_acc = 0
                    num_batches = 0
                    total_auc = 0
                    y_pred = []
                    y_pred_proba = []
                    y_shuffled = []
                    total_batch_acc = 0
                    #Anamika added code for error analysis
                    y_pred_proba_pos = []
                    y_pred_proba_neg = []
                    for ii, (x, y) in enumerate(batch_generator(x_dev, y_dev, batch_size, Trainable=False), 1):
                        feed_dict = {cnn.input_x: x, cnn.input_y: y, cnn.dropout_keep_prob: 1.0}
                        #loss, accuracy, auc = sess.run([cnn.loss, cnn.accuracy, cnn.auc],feed_dict)
                       # batch_pred,batch_pred_proba,loss, accuracy  = sess.run([cnn.loss, cnn.accuracy],feed_dict)
                        batch_pred,batch_pred_proba,loss, accuracy  = sess.run([cnn.predictions, cnn.pred_proba, cnn.loss, cnn.accuracy],feed_dict)
                        total_loss += loss*len(x)
                        total_acc += accuracy*len(x)
                        
                        batch_accuracy= np.sum(y==batch_pred)/y.shape[0]
                        total_batch_acc += batch_accuracy
                        y_pred= np.concatenate([y_pred, batch_pred])
                        y_pred_proba= np.concatenate([y_pred_proba, batch_pred_proba[:,1]])
                        y_shuffled = np.concatenate([y_shuffled, y])
                        y_pred_proba_pos= np.concatenate([y_pred_proba_pos, batch_pred_proba[:,1]])
                        y_pred_proba_neg= np.concatenate([y_pred_proba_neg, batch_pred_proba[:,0]])
                        
                        num_batches += 1
                        
                    avg_loss = total_loss/(num_batches*batch_size)
                    avg_acc = total_acc/(num_batches*batch_size)
                                    
                    #right_acc = total_batch_acc/(num_batches)
                    #avg_auc = total_auc/(num_batches*batch_size)
                    
                    #Calculate Accuracy
                    #new_acc = accuracy_score(y_shuffled, y_pred, normalize=True ) 
                     
                    false_pos_rate, true_pos_rate, _ = roc_curve(y_shuffled, y_pred_proba)  
                    roc_auc = auc(false_pos_rate, true_pos_rate)
                    f1_pos = f1_score(y_shuffled, y_pred, average = None)[1]
                    f1_neg = f1_score(y_shuffled, y_pred, average = None)[0]
                    f1_avg = f1_score(y_shuffled, y_pred, average = 'macro')
                    
                    results['acc'][e] = avg_acc
                    results['auc'][e] = roc_auc
                    results['f1_avg'][e] = f1_avg
                    results['f1_pos'][e] = f1_pos
                    results['f1_neg'][e] = f1_neg
                    
                    
                #time_str = datetime.datetime.now().isoformat()
                    print("\t\t",tkey,"Dev epoch %d, average loss %0.3f,average accuracy %0.3f,auc %0.3f,f1_pos %0.3f,f1_neg %0.3f,f1_avg %0.3f"
                  %(e, avg_loss, avg_acc, roc_auc,f1_pos,f1_neg,f1_avg))
                    #print("\t\tDev epoch {}, auc {:g}, new accuracy {:g}, right accuracy {:g},".format(e,  roc_auc, new_acc, right_acc))
                    #print("\t\tDev epoch {}, average loss {:g}, average accuracy {:g},average auc {:g}".format(e, avg_loss, avg_acc, avg_auc))
                if e%time_print == 0:
                    end = time.time()
                    print("\t\t\t\t    Time taken for",e,"epochs = ", end-start)
                    
                    
                  ##Anamika added code 
            
              #for error analysis
        print("ERROR ANALYSIS")
        print(len(y_shuffled))
        src_key = skey
        tar_key = tkey
        print("src_key", skey, "tar_key", tkey)
        pos_err_pos = np.where((y_shuffled != y_pred) & (y_shuffled ==1))
        neg_err_pos = np.where((y_shuffled != y_pred) & (y_shuffled ==0))
        
        
        
        #for actual negatives that model predicted negatives(true negatives)
        print("True negatives")
        #no_err_neg_probas = y_pred_proba_neg[np.where((y_shuffled == y_pred) & (y_shuffled ==0))]
        no_err_neg_positions = np.where((y_shuffled == y_pred) & (y_shuffled ==0))
        no_err_neg_probas = y_pred_proba_neg[no_err_neg_positions]
        print("Values in no_err_neg_probas", len(no_err_neg_probas))
        #neg_err_pos_vals = 
        #if y_shuffled ==0 and y_pred == 1:
        print("Correct neg probabilities > 0.9", len(no_err_neg_probas[no_err_neg_probas >= 0.9]))
        pos_nine_buckets = np.where((y_shuffled == y_pred) & (y_shuffled ==0) & (y_pred_proba_neg >= 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_nine_buckets[0][n]], "Pred y", y_pred[pos_nine_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_nine_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_nine_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_nine_buckets[0][n]]))
            
            print(dict_dev_df[t_key].reviewText.iloc[pos_nine_buckets[0][n]],'\n')
        
        print("Correct neg probabilities between 0.8 and 0.9", len(no_err_neg_probas[no_err_neg_probas >= 0.8 ])- len(no_err_neg_probas[no_err_neg_probas >= 0.9]))
        pos_eight_buckets = np.where((y_shuffled == y_pred) & (y_shuffled ==0) & (y_pred_proba_neg >= 0.8) & (y_pred_proba_neg < 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_eight_buckets[0][n]], "Pred y", y_pred[pos_eight_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_eight_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_eight_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_eight_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_eight_buckets[0][n]],'\n')
        print("Correct neg probabilities between 0.7 and 0.8", len(no_err_neg_probas[no_err_neg_probas >= 0.7 ])- len(no_err_neg_probas[no_err_neg_probas >= 0.8]))
        pos_seven_buckets = np.where((y_shuffled == y_pred) & (y_shuffled ==0) & (y_pred_proba_neg >= 0.7) & (y_pred_proba_neg < 0.8))
        
        for n in range(2):
            print("actual y", y_shuffled[pos_seven_buckets[0][n]], "Pred y", y_pred[pos_seven_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_seven_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_seven_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_seven_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_eight_buckets[0][n]],'\n')
        print("Correct neg probabilities between 0.6 and 0.7", len(no_err_neg_probas[no_err_neg_probas >= 0.6 ])- len(no_err_neg_probas[no_err_neg_probas >= 0.7]))
        print("Correct neg probabilities between 0.5 and 0.6", len(no_err_neg_probas[no_err_neg_probas >= 0.5 ])- len(no_err_neg_probas[no_err_neg_probas >= 0.6]))
        print("Correct neg probabilities < 0.5", len(no_err_neg_probas[no_err_neg_probas <0.5 ]))
        
        print("")
        print("")
        
        #for actual negatives that model predicted positives(False positives)
        print("False positives")
        neg_err_pos_probas = y_pred_proba_pos[neg_err_pos]
        print("Values in neg_err_pos_probas", len(neg_err_pos_probas))
        #neg_err_pos_vals = 
        #if y_shuffled ==0 and y_pred == 1:
        print("Pos probabilities > 0.9", len(neg_err_pos_probas[neg_err_pos_probas >= 0.9]))
        pos_nine_buckets = np.where((y_shuffled != y_pred) & (y_shuffled ==0) & (y_pred_proba_pos >= 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_nine_buckets[0][n]], "Pred y", y_pred[pos_nine_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_nine_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_nine_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_nine_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_nine_buckets[0][n]],'\n')
        
        print("Pos probabilities between 0.8 and 0.9", len(neg_err_pos_probas[neg_err_pos_probas >= 0.8 ])- len(neg_err_pos_probas[neg_err_pos_probas >= 0.9]))
        pos_eight_buckets = np.where((y_shuffled != y_pred) & (y_shuffled ==0) & (y_pred_proba_pos >= 0.8) & (y_pred_proba_pos < 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_eight_buckets[0][n]], "Pred y", y_pred[pos_eight_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_eight_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_eight_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_eight_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_eight_buckets[0][n]],'\n')
            
        print("Pos probabilities between 0.7 and 0.8", len(neg_err_pos_probas[neg_err_pos_probas >= 0.7 ])- len(neg_err_pos_probas[neg_err_pos_probas >= 0.8]))
        pos_seven_buckets = np.where((y_shuffled != y_pred) & (y_shuffled ==0) & (y_pred_proba_pos >= 0.7) & (y_pred_proba_pos < 0.8))
        for n in range(2):
            print("actual y", y_shuffled[pos_seven_buckets[0][n]], "Pred y", y_pred[pos_seven_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_seven_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_seven_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_seven_buckets[0][n]]))
            print(dict_dev_df[tar_key].reviewText.iloc[pos_seven_buckets[0][n]],'\n')
        print("Pos probabilities between 0.6 and 0.7", len(neg_err_pos_probas[neg_err_pos_probas >= 0.6 ])- len(neg_err_pos_probas[neg_err_pos_probas >= 0.7]))
        print("Pos probabilities between 0.5 and 0.6", len(neg_err_pos_probas[neg_err_pos_probas >= 0.5 ])- len(neg_err_pos_probas[neg_err_pos_probas >= 0.6]))
        print("Pos probabilities < 0.5", len(neg_err_pos_probas[neg_err_pos_probas <0.5 ]))
        
        print("")
        print("")
        
        
        
        
  #for actual positives that model predicted positives(true positives)
        print("True positives")
        #no_err_neg_probas = y_pred_proba_neg[np.where((y_shuffled == y_pred) & (y_shuffled ==0))]
        no_err_pos_positions = np.where((y_shuffled == y_pred) & (y_shuffled ==1))
        no_err_pos_probas = y_pred_proba_pos[no_err_pos_positions]
        print("Values in no_err_pos_probas", len(no_err_pos_probas))
        #neg_err_pos_vals = 
        #if y_shuffled ==0 and y_pred == 1:
        print("Correct positivr probabilities > 0.9", len(no_err_pos_probas[no_err_pos_probas >= 0.9]))
        pos_nine_buckets = np.where((y_shuffled == y_pred) & (y_shuffled ==1) & (y_pred_proba_pos >= 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_nine_buckets[0][n]], "Pred y", y_pred[pos_nine_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_nine_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_nine_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_nine_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_nine_buckets[0][n]],'\n')
        
        print("Correct pos probabilities between 0.8 and 0.9", len(no_err_pos_probas[no_err_pos_probas >= 0.8 ])- len(no_err_pos_probas[no_err_pos_probas >= 0.9]))
        pos_eight_buckets = np.where((y_shuffled == y_pred) & (y_shuffled ==1) & (y_pred_proba_pos >= 0.8) & (y_pred_proba_pos < 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_eight_buckets[0][n]], "Pred y", y_pred[pos_eight_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_eight_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_eight_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_eight_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_eight_buckets[0][n]],'\n')
        print("Correct pos probabilities between 0.7 and 0.8", len(no_err_pos_probas[no_err_pos_probas >= 0.7 ])- len(no_err_pos_probas[no_err_pos_probas >= 0.8]))
        pos_seven_buckets = np.where((y_shuffled == y_pred) & (y_shuffled ==1) & (y_pred_proba_pos >= 0.7) & (y_pred_proba_pos < 0.8))
        for n in range(2):
            print("actual y", y_shuffled[pos_seven_buckets[0][n]], "Pred y", y_pred[pos_seven_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_seven_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_seven_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_seven_buckets[0][n]]))
            print(dict_dev_df[tar_key].reviewText.iloc[pos_seven_buckets[0][n]],'\n')
        print("Correct pos probabilities between 0.6 and 0.7", len(no_err_pos_probas[no_err_pos_probas >= 0.6 ])- len(no_err_pos_probas[no_err_pos_probas >= 0.7]))
        print("Correct pos probabilities between 0.5 and 0.6", len(no_err_pos_probas[no_err_pos_probas >= 0.5 ])- len(no_err_pos_probas[no_err_pos_probas >= 0.6]))
        print("Correct pos probabilities < 0.5", len(no_err_pos_probas[no_err_pos_probas <0.5 ]))
        
        print("")
        print("")    
        
        
#for actual positives that model predicted as negatives(False negatives)y_pred is 0

        
        print("False negatives")
        pos_err_neg_probas = y_pred_proba_neg[pos_err_pos]
        print("Values in pos_err_neg_probas", len(pos_err_neg_probas))
        #neg_err_pos_vals = 
        #if y_shuffled ==0 and y_pred == 1:
        print("Neg probabilities > 0.9", len(pos_err_neg_probas[pos_err_neg_probas >= 0.9]))
        pos_nine_buckets = np.where((y_shuffled != y_pred) & (y_shuffled ==1) & (y_pred_proba_neg >= 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_nine_buckets[0][n]], "Pred y", y_pred[pos_nine_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_nine_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_nine_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_nine_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_nine_buckets[0][n]],'\n')
        
        print("Neg probabilities between 0.8 and 0.9", len(pos_err_neg_probas[pos_err_neg_probas >= 0.8 ])- len(pos_err_neg_probas[pos_err_neg_probas >= 0.9]))
        pos_eight_buckets = np.where((y_shuffled != y_pred) & (y_shuffled ==1) & (y_pred_proba_neg > 0.8) & (y_pred_proba_neg < 0.9))
        for n in range(5):
            print("actual y", y_shuffled[pos_eight_buckets[0][n]], "Pred y", y_pred[pos_eight_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_eight_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_eight_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_eight_buckets[0][n]]))
            
            print(dict_dev_df[tar_key].reviewText.iloc[pos_eight_buckets[0][n]],'\n')
        print("Pos probabilities between 0.7 and 0.8", len(pos_err_neg_probas[pos_err_neg_probas >= 0.7 ])- len(pos_err_neg_probas[pos_err_neg_probas >= 0.8]))
        pos_seven_buckets = np.where((y_shuffled != y_pred) & (y_shuffled ==1) & (y_pred_proba_neg > 0.7) & (y_pred_proba_neg < 0.8))
        for n in range(2):
            print("actual y", y_shuffled[pos_seven_buckets[0][n]], "Pred y", y_pred[pos_eight_buckets[0][n]])
            print('Pos prob value', y_pred_proba_pos[pos_seven_buckets[0][n]])
            print('Neg prob value', y_pred_proba_neg[pos_seven_buckets[0][n]])
            print('review length', np.count_nonzero(dict_dev_ids[tar_key][pos_seven_buckets[0][n]]))
            print(dict_dev_df[tar_key].reviewText.iloc[pos_seven_buckets[0][n]],'\n')
        print("Pos probabilities between 0.6 and 0.7", len(pos_err_neg_probas[pos_err_neg_probas >= 0.6 ])- len(pos_err_neg_probas[pos_err_neg_probas >= 0.7]))
        print("Pos probabilities between 0.5 and 0.6", len(pos_err_neg_probas[pos_err_neg_probas >= 0.5 ])- len(pos_err_neg_probas[pos_err_neg_probas >= 0.6]))
        print("Pos probabilities < 0.5", len(pos_err_neg_probas[pos_err_neg_probas <0.5 ]))
           
                    
                    
                    
                    
                    
    return results
                    

#Function to calculate the predicted probability for positive and negative class on target train set using source model
#source model is the one built on the combined vocabulary of both
def predict_transfer_probability(src_key, size, tar_key):
    #size here is full target train set size - so we can calculate uncertainty on all of it before sorting
    
    batch_size=50
    print('target',tar_key,'source', src_key)
    V = len(dict_transfer_vect[src_key][tar_key].vocabulary_)
    
    size_folder =  "size_" + str(size) 
    out_dir  = os.path.abspath(os.path.join(os.path.curdir, "runs", src_key, tar_key, size_folder))
    #out_dir  = os.path.abspath(os.path.join(os.path.curdir, "runs", src_key))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))    
    print(checkpoint_dir)
    src_model = ''.join([src_key, tar_key])
    #graph_meta_file = checkpoint_dir + '/' + 'hnk01_model.meta'
    graph_meta_file = checkpoint_dir + '/' + src_model +'_model.meta'
    graph=tf.Graph()
    
    x_train = dict_transfer_train_ids[tar_key][src_key][:size]
    y_train = dict_train_y[tar_key][:size]

    with graph.as_default():
        with tf.Session() as sess:
    
      #new_saver = tf.train.import_meta_graph(checkpoint_dir/'hnk_model.meta')
            new_saver = tf.train.import_meta_graph(graph_meta_file)
            new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        
            #create graph from saved model
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        
            pred_proba = graph.get_operation_by_name("output/pred_proba").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        
            y_pred = []
            y_pred_proba = []
            total_batch_acc = 0
            num_batches = 0
            y_shuffled = []
            abs_y_pred_proba = []
            for ii, (x, y) in enumerate(batch_generator(x_train, y_train, batch_size, Trainable=False), 1):
                        
                feed_dict = {input_x: x, input_y: y, dropout_keep_prob: 1.0}
                batch_pred, batch_pred_proba  = sess.run([ predictions, pred_proba],feed_dict)
                batch_accuracy= np.sum(y==batch_pred)/y.shape[0]
                total_batch_acc += batch_accuracy
                y_pred= np.concatenate([y_pred, batch_pred])
                y_pred_proba= np.concatenate([y_pred_proba, batch_pred_proba[:,1]])
                abs_y_pred_proba = np.concatenate([abs_y_pred_proba,np.absolute(batch_pred_proba[:,1] - batch_pred_proba[:,0])])
                y_shuffled = np.concatenate([y_shuffled, y])

                num_batches += 1       
            #y_pred = np.array(y_pred_list)         
              
            new_acc = total_batch_acc/(num_batches)
            print(new_acc)
        
            # Calculate auc
            # false_pos_rate, true_pos_rate, _ = roc_curve(y_dev, y_pred_proba[:,1])
            false_pos_rate, true_pos_rate, _ = roc_curve(y_shuffled, y_pred_proba)  
            roc_auc = auc(false_pos_rate, true_pos_rate)
            print(src_key, tar_key, "AUC","{:.02%}".format(roc_auc))
            
            #Calculate Accuracy
            acc = accuracy_score(y_shuffled, y_pred, normalize=True )
            #print(np.sum(y_shuffled==y_pred)/y_pred.shape[0])
            print('source',src_key, 'target',tar_key, "accuracy","{:.02%}".format(acc))
            print("")
        
        #Save absolute_y_pred_proba
        
        #check if the batching process left remainders. This will result in incorrect length of y_pred_proba saved
        if y_train.shape[0] != abs_y_pred_proba.shape[0]:
            print("Length of y_pred_proba does not match y_dev. Fix batch_size")
            print("Pred proba file not saved")
        else:
            return abs_y_pred_proba
#             file_name = "src_" + src_key + "_tar_" + tar_key + "_" + "train" + str(y_train.shape[0])
#             np.savez_compressed(file_name,pred_prob=abs_y_pred_proba)
#             print( file_name, "Saved file successfully")

#Continue training with adding random samples from target domain
num_epochs = 15
size_model = size_initial
size_list = [150000]
src_key = s_key
tgt_key = t_key
print('Source domain',src_key,'Target Domain',tgt_key)
results_random = pd.DataFrame()
for size in size_list:
    print('Training on target sample of size:',size)
    tgt_train_df = dict_transfer_train_ids[tgt_key][src_key][:size]
    tgt_train_y = dict_train_y[tgt_key][:size]
    print(tgt_train_df.shape,tgt_train_y.shape)
    results = continue_transfer_train(src_key,size_model,tgt_key,tgt_train_df,tgt_train_y)
    results['size'] = size
    results_random = pd.concat([results_random,results])

results_random

print(s_key)
print(t_key)

#Calculating certainty for target train set review

src_key = s_key #source domain
tgt_key = t_key #target domain
size = size_initial #This is source model train set data size to read from the right file.

#calculate absolute difference of positive and negative class probability for target train set using source model built on combined vocab.

u_train_target_abs = predict_transfer_probability(src_key, size, tgt_key)
train_target_len = np.count_nonzero(dict_transfer_train_ids[tgt_key][src_key],axis =1)
c_div_len_target = u_train_target_abs*150/train_target_len #calculates certainty per word id in the review
# file_name = "src_" + src_key + "_tar_" + tar_key + "_" + "train" + str(size)
# u_train_target_abs = np.load(file_name)
print(u_train_target_abs.shape,train_target_len.shape,c_div_len_target.shape)
print('max, min uncertainty absolute',np.max(u_train_target_abs),np.min(u_train_target_abs))
print('max, min length',np.max(train_target_len),np.min(train_target_len))
print('max, min certainty*length',np.max(c_div_len_target),np.min(c_div_len_target))

#See if certainty is correlated with length
span = 0.1
sort_ids = np.argsort(u_train_target_abs)
u_train_target_sorted = u_train_target_abs[sort_ids]
train_target_len_sorted = train_target_len[sort_ids]
c_div_len_sorted = c_div_len_target[sort_ids]
range_l = int(span*len(u_train_target_sorted))

for i in range(np.int(1/span)):
    print("For range %0.2f to %0.2f, average certainty = %0.2f, average length = %0.2f, average certainty per length = %0.2f"
          %(i*span,(i+1)*span,np.average(u_train_target_sorted[i*range_l:(i+1)*range_l]),np.average(train_target_len_sorted[i*range_l:(i+1)*range_l]),
           np.average(c_div_len_sorted[i*range_l:(i+1)*range_l])))

import matplotlib.pyplot
import pylab

matplotlib.pyplot.scatter(u_train_target_abs,train_target_len)

matplotlib.pyplot.show()

#Active transfer learning : Continue training with adding selected samples from target domain
#In this cell, samples where we have the least absolute difference in predicted probability of positive and negative class are added first.

num_epochs = 15
size_model = size_initial
src_key = s_key
tgt_key = t_key

#Create a sorted version of the certainty, and correspondingly sorted target train set ids, and labels.
sort_ids = np.argsort(u_train_target_abs)
certainty_sorted = u_train_target_abs[sort_ids]
#print(sort_ids)
df_target_ids_pre = dict_transfer_train_ids[tgt_key][src_key]
df_target_labels_pre = dict_train_y[tgt_key]
print('Target labels pre sort',df_target_labels_pre[-20:])
print(type(df_target_labels_pre))
#df_target_ids_pre = df_target_ids_pre.iloc([sort_ids])
df_target_ids = df_target_ids_pre[sort_ids]
df_target_labels = df_target_labels_pre[sort_ids]
print('\n Target labels post sort',df_target_labels[-20:])
print('\n Certainty sorted','\n First 20',certainty_sorted[:20],'\n Last 20',certainty_sorted[-20:])
results_least_certain = pd.DataFrame()


print('\nTraining on least certain first')
size_list = size_list
for size in size_list:
    avg_certainty = np.average(certainty_sorted[:size])
    print('Training on target sample of size:',size,'with average certainty %0.3f'%avg_certainty)
    tgt_train_df = df_target_ids[:size]
    tgt_train_y = df_target_labels[:size]
    avg_certainty = np.average(certainty_sorted[:size])
    print(tgt_train_df.shape,tgt_train_y.shape)
    results = continue_transfer_train(src_key,size_model,tgt_key,tgt_train_df,tgt_train_y)
    results['size'] = size
    results_least_certain = pd.concat([results_least_certain,results])
    
results_least_certain

#In this cell, samples where we have the most absolute difference in predicted probability of positive and negative class are added first.
size_model = size_initial
print('Training on most certain first')
size_list = size_list
results_most_certain = pd.DataFrame()
for size in size_list:
    avg_certainty = np.average(certainty_sorted[-size:])
    print('Training on target sample of size:',size,'with average certainty %0.3f'%avg_certainty)
    tgt_train_df = df_target_ids[-size:]
    tgt_train_y = df_target_labels[-size:]
    print(tgt_train_df.shape,tgt_train_y.shape)
    results = continue_transfer_train(src_key,size_model,tgt_key,tgt_train_df,tgt_train_y)
    results['size'] = size
    results_most_certain = pd.concat([results_most_certain,results])
    
results_most_certain
   

#In this cell, we add samples with the lowest certainty per word id first.
size_model = size_initial
src_key = s_key
tgt_key = t_key

#Create a sorted version of the certainty, and correspondingly sorted target train set ids, and labels.
sort_ids = np.argsort(c_div_len_target)
c_div_len_sorted = c_div_len_target[sort_ids]
certainty_sorted = u_train_target_abs[sort_ids]
#print(sort_ids)
df_target_ids_pre = dict_transfer_train_ids[tgt_key][src_key]
df_target_labels_pre = dict_train_y[tgt_key]
print('Target labels pre sort',df_target_labels_pre[-20:])
print(type(df_target_labels_pre))
#df_target_ids_pre = df_target_ids_pre.iloc([sort_ids])
df_target_ids = df_target_ids_pre[sort_ids]
df_target_labels = df_target_labels_pre[sort_ids]
print('\n Target labels post sort',df_target_labels[-20:])
print('\n Certainty sorted','\n First 20',certainty_sorted[:20],'\n Last 20',certainty_sorted[-20:])

results_cperlen = pd.DataFrame()


print('\nTraining on least certain first')
size_list = size_list
for size in size_list:
    avg_certainty = np.average(certainty_sorted[:size])
    avg_c_div_len = np.average(c_div_len_sorted[:size])
    print('Training on target sample of size:',size,'with average certainty per word id',avg_c_div_len,'with average certainty %0.3f'%avg_certainty)
    tgt_train_df = df_target_ids[:size]
    tgt_train_y = df_target_labels[:size]
    avg_certainty = np.average(certainty_sorted[:size])
    print(tgt_train_df.shape,tgt_train_y.shape)
    results = continue_transfer_train(src_key,size_model,tgt_key,tgt_train_df,tgt_train_y)
    results['size'] = size
    results_cperlen = pd.concat([results_cperlen,results])
    
results_cperlen

