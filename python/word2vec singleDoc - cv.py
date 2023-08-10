import numpy as np
from gensim.models import Word2Vec
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
import ast
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from __future__ import print_function
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from functools import reduce
import os
from os.path import basename
import csv

sequence_length = 1000

# Parameters
# ==================================================
#
# Model Variations. See Kim Yoon’s Convolutional Neural Networks for 
# Sentence Classification, Section 3 for detail.

model_variation = 'CNN-google'  #  CNN-rand | CNN-google
print('Model variation is %s' % model_variation)

# Model Hyperparameters
embedding_dim = 300
filter_sizes = (3, 4)
num_filters = 3
dropout_prob = (0.25, 0.5)
hidden_dims = 100

# Training parameters
batch_size = 5
num_epochs = 5
#val_split = 0.33

# Word2Vec parameters, see train_word2vec
min_word_count = 1  # Minimum word count                        
context = 4        # Context window size    
ACTION = "train"
weights_file = "weights_file"
TEXT_DATA_DIR='SingleDocSignals'

# Data Preparatopn
# ==================================================
#
# Load data
print("Loading data...")



# Load 10-folds indicies produced from R (Avg_Accuracy:0.7222222)
cv_files=[] 
with open('SingleDocSignals_R_cv_files','r') as intputFile:
        reader=csv.reader(intputFile,delimiter=' ')
        for row in reader:
            cv_files.append(row)
accuracy=[]
allFiles=sorted(os.listdir(TEXT_DATA_DIR))
for fold in range(0, len(cv_files)):
    # load data
    
    train_sentences = []  # list of text articles
    train_labels_index = {}  # dictionary mapping label name to numeric id
    train_labels = []  # list of label ids
    test_sentences = []  # list of text articles
    test_labels_index = {}  # dictionary mapping label name to numeric id
    test_labels = []  # list of label ids
    for fname in allFiles:
        fpath = os.path.join(TEXT_DATA_DIR, fname)
        f = open(fpath)
        if(fname in cv_files[fold]):
            test_sentences.append(f.read())
            test_labels_index[basename(fname)] = len(test_labels_index)
            test_labels.append(-1)
        else:
            train_sentences.append(f.read())
            train_labels_index[basename(fname)] = len(train_labels_index)
            train_labels.append(-1)
                
        f.close()
        

    d = []
    for i in train_sentences:
        words2 = text_to_word_sequence(i, lower=True, split=" ")
        d.append(words2)

    train_sentences = d
    
    train_vocab = sorted(reduce(lambda x, y: x | y, (set(i) for i in d)))
    
    d = []
    for i in test_sentences:
        words2 = text_to_word_sequence(i, lower=True, split=" ")
        d.append(words2)

    test_sentences = d
    
    test_vocab = sorted(reduce(lambda x, y: x | y, (set(i) for i in d)))
    
    # Reserve 0 for masking via pad_sequences
    train_vocab_size = len(train_vocab) + 1
    train_word_idx = dict((c, i + 1) for i, c in enumerate(train_vocab))
    
    X = []
    for i in train_sentences:
        x = [train_word_idx[w] for w in i]
        X.append(x)

    X_train = pad_sequences(X,sequence_length)
    
    test_vocab_size = len(test_vocab) + 1
    test_word_idx = dict((c, i + 1) for i, c in enumerate(test_vocab))
    X = []
    for i in test_sentences:
        x = [test_word_idx[w] for w in i]
        X.append(x)

    X_test = pad_sequences(X,sequence_length)

    #load labels
    filePath='SingleDocSignals.csv'
    with open(filePath,'r') as intputFile:
            reader=csv.reader(intputFile,delimiter=',')
            for fname,y in reader:
                if((fname+".txt") in cv_files[fold]):
                    test_labels[test_labels_index[fname+".txt"]]=int(y)
                else:
                    train_labels[train_labels_index[fname+".txt"]]=int(y)

    Categories = train_labels
    y = np.zeros(9)
    outputs = list(set(Categories))
    Y = []
    for i in Categories:
        y = np.zeros(9)
        indexV = outputs.index(i)
        y[indexV]=1
        Y.append(y)
    train_Y = np.asarray(Y)
    
    train_vocabulary= train_word_idx
    train_vocabulary_inv = train_vocab 
    train_vocabulary_inv.append("</PAD>")
    
    Categories = test_labels
    y = np.zeros(9)
    outputs = list(set(Categories))
    Y = []
    for i in Categories:
        y = np.zeros(9)
        indexV = outputs.index(i)
        y[indexV]=1
        Y.append(y)
    test_Y = np.asarray(Y).argmax(axis=1)
     
    test_vocabulary= test_word_idx
    test_vocabulary_inv = test_vocab 
    test_vocabulary_inv.append("</PAD>")
    
    if model_variation=='CNN-google':
        model_name='GoogleNews-vectors-negative300.bin'
        embedding_model = Word2Vec.load_word2vec_format(model_name, binary=True)
        embedding_weights = [np.array([embedding_model[w] if w in embedding_model                                                        else np.random.uniform(-0.25,0.25,embedding_model.vector_size)                                                        for w in train_vocabulary_inv])]
    elif model_variation=='CNN-rand':
        embedding_weights = None
    else:
        raise ValueError('Unknown model variation')    
     
    
    print("Number of training documents: {:d}".format(len(X_train)))
    print("Vocabulary Size: {:d}".format(len(train_vocabulary)))
    
    # find out how vocab is causing problems
    
    
    # Building model
    # ==================================================
    #
    # graph subnet with one input and one output,
    # convolutional layers concateneted in parallel
    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
    
    if len(filter_sizes) > 1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]
    
    graph = Model(input=graph_in, output=out)
    # main sequential model
    model = Sequential()
    if not model_variation=='CNN-static':
        model.add(Embedding(len(train_vocabulary_inv),embedding_dim, input_length=sequence_length,
                            weights=embedding_weights))
    
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(9))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    # Training the model
    
    model.fit(X_train, train_Y, batch_size=batch_size,nb_epoch=num_epochs, verbose=0)
   
    # validate 
    test_count=len(X_test)
    probs = model.predict(X_test.reshape(test_count, -1))
    miss=np.sum( probs.argmax(axis=1)==test_Y)

    acc=(test_count-miss)/test_count
    accuracy.append(acc)
    print("Fold:%i, Missed %i out of %i, Accuracy:%f"%(fold,miss,test_count,acc))
    
print(accuracy)
print("Average Accuracy=%f"%np.average(accuracy))

