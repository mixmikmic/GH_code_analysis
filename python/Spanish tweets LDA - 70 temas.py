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

# s = open_session()
# all_tweets_text_es = [t.text for t in s.query(Tweet).all() if t.lang == 'es']

# with open('all_tweets_text_es.json', 'w') as f:
#     json.dump(all_tweets_text_es, f)

with open('all_tweets_text_es.json') as f:
    all_tweets_text_es = json.load(f)
len(all_tweets_text_es)    

# tweets = sample(all_tweets_text_es, 5000)
tweets = all_tweets_text_es

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

# dictionary = gensim.corpora.Dictionary(get_docs(tweets))
# dictionary.filter_extremes(no_below=100, no_above=0.1, keep_n=None)
# dictionary.save("tweets_es.dict")

dictionary = gensim.corpora.Dictionary.load("tweets_es.dict")

# bow = [dictionary.doc2bow(doc) for doc in get_docs(tweets)]
# with open('tweets_es_bow.pickle', 'wb') as f:
#     pickle.dump(bow, f)

with open('tweets_es_bow.pickle', 'rb') as f:
    bow = pickle.load(f)

n_topics = 35
iters = 20
passes = 4

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
model.save("tweets_es_%dtopics.lda" % n_topics)

viz = pyLDAvis.gensim.prepare(model, [v for v in bow], model.id2word)
pickle.dump(viz, open("tweets_es_%d.viz" % n_topics,'wb'))

viz = pickle.load(open("tweets_es_%d.viz" % n_topics,'rb'))
pyLDAvis.display(viz)

# Classify each document into only one, most probable topic,
# get topic counts

doc_topics = model.get_document_topics(bow)
doc_topics_dense = np.empty((len(corpus), n_topics))
for i in range(len(corpus)):
    dt = np.zeros(n_topics)
    for t in doc_topics[i]:
        dt[t[0]] = t[1]
    doc_topics_dense[i,:] = dt
labels = np.argmax(doc_topics_dense, axis=1)
labels_unique = dict(zip(
    np.unique(labels, return_counts=True)[0],
    np.unique(labels, return_counts=True)[1]
    ))
labels_unique = sorted(labels_unique.items(), key=operator.itemgetter(1), reverse=True)
labels_unique

