import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# fixes weird issue with pyLDAvis (?) warnings

import pandas as pd
import numpy as np
import pickle
import operator
import re
import gc
import gensim
# from gensim.similarities import WmdSimilarity

import pyLDAvis
import pyLDAvis.gensim

# wtf
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from tw_dataset.dbmodels import *
from random import sample
import json

from create_lda_datasets import *

# s = open_session()
# all_tweets_text_es = [t.text for t in s.query(Tweet).all() if t.lang == 'es']

# with open('all_tweets_text_es.json', 'w') as f:
#     json.dump(all_tweets_text_es, f)

f1s = load_nlp_selected_users()

uid, f1 = f1s[4]

uid = int(uid)

from experiments.datasets import load_dataframe

Xtrain_fname = join(DATAFRAMES_FOLDER, "dfXtrain_%d.pickle" % uid)
Xtest_fname = join(DATAFRAMES_FOLDER, "dfXtestv_%d.pickle" % uid)
Xvalid_fname = join(DATAFRAMES_FOLDER, "dfXvalid_%d.pickle" % uid)

X_train = pd.read_pickle(Xtrain_fname)
X_test = pd.read_pickle(Xtest_fname)
X_valid = pd.read_pickle(Xvalid_fname)

with open('all_tweets_text_es.json') as f:
    all_tweets_text_es = json.load(f)
len(all_tweets_text_es)    

twids = list(X_train.index) + list(X_valid.index) + list(X_test.index)





s = open_session()

# tweets = sample(all_tweets_text_es, 5000)
tweets = s.query(Tweet.text).filter(Tweet.id.in_(twids)).all()

tweets = [t[0] for t in tweets]

len(tweets)

from tokenizer import tokenize, spanish_stopwords

def preprocess(doc):
    pre_doc = doc
        
    # remove URLs
    pre_doc = re.sub(
        r"https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        " ", pre_doc)
        
    # find and split hashtags
    # very simple splitting (TODO: come up with something wittier)
    # split on capital letters, but only if hashtag longer than 5
    # → conditional is to avoid splitting abbr. like "IoT" or "NSA"
    pre_doc = re.sub(r"(?:^|\s)[＃#]{1}(\w+)", 
            lambda s: re.sub(r"([A-Z])", r" \1", s.group(0)) if len(s.group(0)) > 5 else s.group(0), 
            pre_doc)
    pre_doc = re.sub(r"＃|#", " ", pre_doc)
    
    # lowercase everything
    pre_doc = pre_doc.lower()
        
    # remove bullshit
    pre_doc = re.sub(r"\@|\'|\"|\\|…|\/|\-|\||\(|\)|\.|\,|\!|\?|\:|\;|“|”|’|—", " ", pre_doc)
    
    # normalize whitespaces
    pre_doc = re.sub(r"\s+", " ", pre_doc)
    pre_doc = re.sub(r"(^\s)|(\s$)", "", pre_doc)
    
    return pre_doc

class get_docs(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for doc in self.corpus:
            tokens = tokenize(preprocess(doc))
            yield tokens

dictionary = gensim.corpora.Dictionary(get_docs(tweets))
dictionary.filter_extremes(no_below=100, no_above=0.1, keep_n=None)
dictionary.save("tweets_es_%d.dict" % uid)

dictionary = gensim.corpora.Dictionary.load("tweets_es_%d.dict" % uid)

bow = [dictionary.doc2bow(doc) for doc in get_docs(tweets)]
with open('tweets_es_bow.pickle', 'wb') as f:
    pickle.dump(bow, f)

with open('tweets_es_bow.pickle', 'rb') as f:
    bow = pickle.load(f)

n_topics = 10
iters = 100
passes = 10

# gensim.models.ldamodel.LdaModel?

# model = gensim.models.ldamodel.LdaModel(
model = gensim.models.LdaMulticore(
        corpus=bow,
        id2word=dictionary,
        num_topics=n_topics,
        iterations=iters,
        alpha=0.001,
        passes=passes,
        chunksize=10000,
        workers=7
#         distributed=True
)
model.save("tweets_es_u%d_%dtopics.lda" % (uid, n_topics))

model = gensim.models.LdaModel.load("tweets_es_u%d_%dtopics.lda" % (uid, n_topics))

viz = pyLDAvis.gensim.prepare(model, [v for v in bow], model.id2word)
pickle.dump(viz, open("tweets_es_u%d_%d.viz" % (uid, n_topics),'wb'))

viz = pickle.load(open("tweets_es_u%d_%d.viz" % (uid, n_topics),'rb'))
pyLDAvis.display(viz)

X_train, X_valid, X_test, y_train, y_valid, y_test = load_small_validation_dataframe(uid)

X_train_inds = X_train.index
X_valid_inds = X_valid.index
X_test_inds = X_test.index

train_tweets = [s.query(Tweet).get(twid) for twid in X_train_inds]
valid_tweets = [s.query(Tweet).get(twid) for twid in X_valid_inds]
test_tweets = [s.query(Tweet).get(twid) for twid in X_test_inds]

def get_lda(t):
    tokens = tokenize(preprocess(t.text))
    doc_bow = dictionary.doc2bow(tokens)
    doc_lda = model[doc_bow]
    
    return doc_lda

rows_train = [get_lda(t) for t in train_tweets]

rows_valid = [get_lda(t) for t in valid_tweets]

rows_test = [get_lda(t) for t in test_tweets]

def rows_to_csc(rows):
    data = []
    row_ind = []
    col_ind = []
    for i, r in enumerate(rows):
        for j, d in r:
            row_ind.append(i)
            col_ind.append(j)
            data.append(d)
    return csc_matrix((data, (row_ind, col_ind)))

X_train_lda = rows_to_csc(list(rows_train))
X_valid_lda = rows_to_csc(list(rows_valid))
X_test_lda = rows_to_csc(list(rows_test))



X_train_combined = sp.hstack((X_train, X_train_lda))
X_valid_combined = sp.hstack((X_valid, X_valid_lda))
X_test_combined = sp.hstack((X_test, X_test_lda))

from experiments._1_one_user_learn_neighbours.classifiers import model_select_svc2

ds_comb = (X_train_combined, X_valid_combined, y_train, y_valid)
comb_clf = model_select_svc2(ds_comb, n_jobs=5)

ds = (X_train, X_valid, y_train, y_valid)
clf = model_select_svc2(ds, n_jobs=5)



