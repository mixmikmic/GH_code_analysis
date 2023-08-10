# File for downloading GloVe and using it as the predictors in a random forest model

import gensim.models
from gensim.models import KeyedVectors
from gensim.models import word2vec
from gensim.models import Word2Vec
import logging
import numpy as np
import pandas as pd
import math

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Read in data
dat = pd.read_excel('all_sentences.xlsx')
dat.head()

# Create dictionary of words in corpus
sentence = dat['words_clean'].iloc[0]
words = sentence.split(' ')

float(words.count(words[0]))/len(words)

word_dict = {}

for sentence in dat['words_clean']:
    for word in sentence.split(' '):
        if word not in word_dict:
            word_dict[word] = sum([1 for sentence in dat['words_clean'] if word in sentence.split(' ')]) 

# Sanity check
word_dict['forcing']

# Implementing IDF by hand
idf = []
for sentence in dat['words_clean']:   
    word_freq = []
    for word in sentence.split(' '):
        word_freq.append(word_dict[word])
    idf.append([math.log(dat.shape[0]/float(count)) for count in word_freq])

idf_dict = {}
for key,val in word_dict.iteritems():
    idf_dict[key] = math.log(dat.shape[0]/float(val))

idf_dict['forcing']

len(idf), dat.shape

# Add IDF values to dataframe
dat['idf'] = idf

dat.head()

word_freq = []
for word in sentence.split(' '):
    word_freq.append(word_dict[word])
    
print [math.log(dat.shape[0]/float(count)) for count in word_freq]
print word_freq
print [float(count)/sum(word_freq) for count in word_freq]

dat['words_clean'].iloc[0]

# MIght not need all this??

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(dat['words_clean'])

bag_of_words

tf_transformer = TfidfTransformer(use_idf=False).fit(bag_of_words)
X_train_tf = tf_transformer.transform(bag_of_words)
X_train_tf.shape

X_train_tf[0,1000]

# Import pretrained glove vectors
from gensim.models.keyedvectors import KeyedVectors
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)

# Testing vectors
glove_model.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)

# Testing vectors
glove_model.most_similar('excited', topn=5)

# Testing vectors
glove_model.wv['daniel']

word_vectors = glove_model.wv

type(word_vectors)

# Create columns for avg_vec, sum_vec, and both of those weighted
avg_vec = np.full(dat.shape[0], None)
sum_vec = np.full(dat.shape[0], None)
weighted_avg_vec = np.full(dat.shape[0], None)
weighted_sum_vec = np.full(dat.shape[0], None)
for i, sentence in enumerate(dat['words_clean']):
    words = sentence.split(' ')
    #print i
    real_words = []
    real_weights = []
    for word in words:
        if word in glove_model.wv:
            real_words.append(word)
            real_weights.append(idf_dict[word])
            
    vecs = np.full(len(real_words), None)
    weighted_vecs = np.full(len(real_words), None)
    #print words
    for j, word in enumerate(real_words):
        vecs[j] = glove_model.wv[word]
        weighted_vecs[j] = glove_model.wv[word] * real_weights[j]
    
    weighted_sum_vec[i] = sum(weighted_vecs)
    weighted_avg_vec[i] = sum(weighted_vecs)/float(sum(real_weights))
    avg_vec[i] = sum(vecs)/len(vecs)
    sum_vec[i] = sum(vecs)
    
dat['sum_vec'] = sum_vec
dat['avg_vec'] = avg_vec
dat['weighted_avg_vec'] = weighted_avg_vec
dat['weighted_sum_vec'] = weighted_sum_vec

dat.head(n=20)

len(dat['idf'].iloc[0])

len(dat['avg_vec'].iloc[0])

# Export
dat.to_excel('dat_large.xlsx')

type(dat['sum_vec'].iloc[0])

from sklearn import ensemble
from sklearn.model_selection import train_test_split

def return_acc(probs, y_test, thresh):
    y_pred = []
    for row in probs:
        if row[1] > thresh:
            y_pred.append(1)
        else:
            y_pred.append(np.argmax(row))
    #print(len(y_pred), len(y_test))
    return y_pred, np.mean(np.array(y_pred) == y_test)

# Tuning neutral threshold
accuracy = [0]*50
for j in range(20):
    X_train, X_test, y_train, y_test = train_test_split(dat['weighted_avg_vec'], dat['label'], test_size=0.3, random_state=j)
    X_train_mat = np.matrix(X_train.values.tolist())
    X_test_mat = np.matrix(X_test.values.tolist())
    rf_model = ensemble.RandomForestClassifier(max_depth=200, n_estimators=50)
    rf_model.fit(X_train_mat, y_train) 
    threshes = np.arange(0.0, 0.5, 0.01)
    for i, thresh in enumerate(threshes):
        probs = rf_model.predict_proba(X_test_mat)
        accuracy[i] += return_acc(probs, y_test, thresh)[1]

new_acc = [acc/20 for acc in accuracy]
print(new_acc)
np.argmax(new_acc)

# Tuning max_depth
depths = [1,2,3,5,10,30,50,70,100,200,300,500,800,1000,1500]
avg_acc = [None]*len(depths)
for i, depth in enumerate(depths):
    accuracy = [0]*20
    for j in range(len(accuracy)):
        X_train, X_test, y_train, y_test = train_test_split(dat['avg_vec'], dat['label'], test_size=0.3, random_state=j)
        X_train_mat = np.matrix(X_train.values.tolist())
        X_test_mat = np.matrix(X_test.values.tolist())
        rf_model = ensemble.RandomForestClassifier(max_depth=200, n_estimators=50)
        rf_model.fit(X_train_mat, y_train) 
        probs = rf_model.predict_proba(X_test_mat)
        accuracy[j] += return_acc(probs, y_test, 0.30)[1]
    avg_acc[i] = sum(accuracy)/len(accuracy)

print(avg_acc)
np.argmax(avg_acc)

X_train, X_test, y_train, y_test = train_test_split(dat['weighted_avg_vec'], dat['label'], test_size=0.3, random_state=j)
X_train_mat = np.matrix(X_train.values.tolist())
X_test_mat = np.matrix(X_test.values.tolist())
rf_model = ensemble.RandomForestClassifier(max_depth=200, n_estimators=50)
rf_model.fit(X_train_mat, y_train) 
probs = rf_model.predict_proba(X_test_mat)
return_acc(probs, y_test, 0.30)[1]

# Functionalize to see which variable is best
def random_forest(var):
    accuracy = [0]*20
    for j in range(len(accuracy)):
        X_train, X_test, y_train, y_test = train_test_split(dat['avg_vec'], dat['label'], test_size=0.3, random_state=j)
        X_train_mat = np.matrix(X_train.values.tolist())
        X_test_mat = np.matrix(X_test.values.tolist())
        rf_model = ensemble.RandomForestClassifier(max_depth=200, n_estimators=50)
        rf_model.fit(X_train_mat, y_train) 
        probs = rf_model.predict_proba(X_test_mat)
        accuracy[j] += return_acc(probs, y_test, 0.30)[1]
    return sum(accuracy)/len(accuracy)

print random_forest('avg_vec')
print random_forest('sum_vec')
print random_forest('weighted_avg_vec')
print random_forest('weighted_sum_vec')

# All forms of the vector get the same score pretty much, none outperform Naive Bayes



