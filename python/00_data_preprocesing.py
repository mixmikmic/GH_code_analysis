#Imports and paths
from __future__ import print_function

import numpy as np

# GPU path
#data_path='/home/ubuntu/data/training/text/sentiment/aclImdb/'

data_path='/Users/jorge/data/training/text/sentiment/aclImdb/'

# Generator of list of files in a folder and subfolders
import os
import shutil
import fnmatch

def gen_find(filepattern, toppath):
    '''
    Generator with a recursive list of files in the toppath that match filepattern 
    Inputs:
        filepattern(str): Command stype pattern 
        toppath(str): Root path
    '''
    for path, dirlist, filelist in os.walk(toppath):
        for name in fnmatch.filter(filelist, filepattern):
            yield os.path.join(path, name)

#Test
#print(gen_find("*.txt", data_path+'train/pos/').next())

def read_sentences(path):
    sentences = []
    sentences_list = gen_find("*.txt", path)
    for ff in sentences_list:
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    return sentences        

#Test
print(read_sentences(data_path+'train/pos/')[0:2])

print(read_sentences(data_path+'train/neg/')[0:2])

def tokenize(sentences):
    from nltk import word_tokenize
    print( 'Tokenizing...',)
    tokens = []
    for sentence in sentences:
        tokens += [word_tokenize(sentence)]
    print('Done!')

    return tokens

print(tokenize(read_sentences(data_path+'train/pos/')[0:2]))

sentences_trn_pos = tokenize(read_sentences(data_path+'train/pos/'))
sentences_trn_neg = tokenize(read_sentences(data_path+'train/neg/'))
sentences_trn = sentences_trn_pos + sentences_trn_neg

#create the dictionary to conver words to numbers. Order it with most frequent words first
def build_dict(sentences):
#    from collections import OrderedDict

    '''
    Build dictionary of train words
    Outputs: 
     - Dictionary of word --> word index
     - Dictionary of word --> word count freq
    '''
    print( 'Building dictionary..',)
    wordcount = dict()
    #For each worn in each sentence, cummulate frequency
    for ss in sentences:
        for w in ss:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = list(wordcount.values()) # List of frequencies
    keys = list(wordcount) #List of words
    
    sorted_idx = reversed(np.argsort(counts))
    
    worddict = dict()
    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)
    print( np.sum(counts), ' total words ', len(keys), ' unique words')

    return worddict, wordcount


worddict, wordcount = build_dict(sentences_trn)

print(worddict['the'], wordcount['the'])

# 
def generate_sequence(sentences, dictionary):
    '''
    Convert tokenized text in sequences of integers
    '''
    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in ss]

    return seqs

# Create train and test data

#Read train sentences and generate target y
train_x_pos = generate_sequence(sentences_trn_pos, worddict)
train_x_neg = generate_sequence(sentences_trn_neg, worddict)
X_train_full = train_x_pos + train_x_neg
y_train_full = [1] * len(train_x_pos) + [0] * len(train_x_neg)

print(X_train_full[0], y_train_full[0])

#Read test sentences and generate target y
sentences_tst_pos = read_sentences(data_path+'test/pos/')
sentences_tst_neg = read_sentences(data_path+'test/neg/')

test_x_pos = generate_sequence(tokenize(sentences_tst_pos), worddict)
test_x_neg = generate_sequence(tokenize(sentences_tst_neg), worddict)
X_test_full = test_x_pos + test_x_neg
y_test_full = [1] * len(test_x_pos) + [0] * len(test_x_neg)

print(X_test_full[0])
print(y_test_full[0])

#Median length of sentences
print('Median length: ', np.median([len(x) for x in X_test_full]))

max_features = 50000 # Number of most frequent words selected. the less frequent recode to 0
maxlen = 200  # cut texts after this number of words (among top max_features most common words)

#Select the most frequent max_features, recode others using 0
def remove_features(x):
    return [[0 if w >= max_features else w for w in sen] for sen in x]

X_train = remove_features(X_train_full)
X_test  = remove_features(X_test_full)
y_train = y_train_full
y_test = y_test_full

print(X_test[1])

from tensorflow.contrib.keras import preprocessing

# Cut or complete the sentences to length = maxlen
print("Pad sequences (samples x time)")

X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print(X_test[0])

# Shuffle data
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train, random_state=0)

# Export train and test data
np.save(data_path + 'X_train', X_train)
np.save(data_path + 'y_train', y_train)
np.save(data_path + 'X_test',  X_test)
np.save(data_path + 'y_test',  y_test)

# Export worddict
import pickle

with open(data_path + 'worddict.pickle', 'wb') as pfile:
    pickle.dump(worddict, pfile)



