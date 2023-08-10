import numpy as np
import pandas as pd

import os, sys
import json, re
import logging

from nltk import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import xgboost as xgb

# word 2 vector
from gensim.models import word2vec

basepath = os.path.expanduser('~/Desktop/src/Stumbleupon_classification_challenge/')
sys.path.append(os.path.join(basepath, 'src'))

np.random.seed(5)

from data import load_datasets
from models import train_test_split, cross_val_scheme
from helpers import util

# load dataset
train, test, sample_sub = load_datasets.load_dataset()

train['is_news'] = train.is_news.fillna(-999)
test['is_news'] = test.is_news.fillna(-999)

train_json = util.convert_to_json(train.boilerplate)
test_json = util.convert_to_json(test.boilerplate)

def remove_non_alphanumeric(sentence):
    return re.sub(r'[^a-z0-9+]', ' ', sentence.lower())

def concatente_boilerplate_components(bp):
    return ' '.join([remove_non_alphanumeric(text) for k, text in bp.items() if text])

train_json_processed = list(map(concatente_boilerplate_components, train_json))
test_json_processed = list(map(concatente_boilerplate_components, test_json))

def tokenize(sentence, removeStopwords=False):
    if removeStopwords:
        return ' '.join([word for word in word_tokenize(sentence) if word not in ENGLISH_STOP_WORDS])
    else:
        return word_tokenize(sentence)

train_json_tokenized = list(map(tokenize, train_json_processed))
test_json_tokenized = list(map(tokenize, test_json_processed))

train_json_tokenized = np.array(train_json_tokenized)
test_json_tokenized = np.array(test_json_tokenized)

sentences = np.hstack([train_json_tokenized, test_json_tokenized])

# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

# Set values for various parameters
num_features = 500    # Word vector dimensionality                      
min_word_count = 3   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers,             size=num_features, min_count = min_word_count,             window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

# Model's vocab
model.syn0.shape

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
        if counter%1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))
       
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,            num_features)
        counter = counter + 1.
    
    return reviewFeatureVecs

clean_train_sentences = []

for sentence in train_json_processed:
    clean_train_sentences.append(tokenize(sentence, removeStopwords=True))

clean_test_sentences = []

for sentence in test_json_processed:
    clean_test_sentences.append(tokenize(sentence, removeStopwords=True))

trainDataVecs = getAvgFeatureVecs( clean_train_sentences, model, num_features )
testDataVecs = getAvgFeatureVecs( clean_test_sentences, model, num_features )

params = {
    'test_size': 0.2,
    'random_state': 2,
    'stratify': train.is_news
}

itrain, itest = train_test_split.tr_ts_split(len(train), **params)

X_train = trainDataVecs[itrain]
X_test = trainDataVecs[itest]

y_train = train.iloc[itrain].label
y_test = train.iloc[itest].label

print(X_train.shape, X_test.shape)

# train a random forest classifier
est = RandomForestClassifier(n_estimators=75, max_depth=15, n_jobs=-1, random_state=10)
est.fit(X_train, y_train)

y_preds = est.predict_proba(X_test)[:, 1]
print('AUC score on the test set: %f' %(roc_auc_score(y_test, y_preds)))



