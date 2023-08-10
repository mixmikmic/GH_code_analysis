import numpy as np
import pandas as pd
import cPickle as pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nlp_utils import LemmaTokenizer, Bigram
from nlp_utils import read_docs, get_topic_words


TEXT_DATA_DIR = '/home/ubuntu/working/text_classification/20_newsgroup/'
MAX_NB_WORDS = 20000
NB_TOPICS = 10

# Read documents
docs, doc_classes = read_docs(TEXT_DATA_DIR)

# Prepocess based on perplexity results
tokenizer = LemmaTokenizer()
phraser = Bigram()
token_docs = [tokenizer(doc) for doc in docs]
bigram_docs = phraser(token_docs)
vectorizer = CountVectorizer(
    min_df=10, max_df=0.5,
    max_features=MAX_NB_WORDS,
    preprocessor = lambda x: x,
    tokenizer = lambda x: x)
corpus = vectorizer.fit_transform(bigram_docs)

# Build model and fit
lda_model = LatentDirichletAllocation(
    n_components=NB_TOPICS,
    learning_method='online',
    max_iter=10,
    batch_size=2000,
    verbose=1,
    max_doc_update_iter=100,
    n_jobs=-1,
    random_state=0)
lda = lda_model.fit(corpus)
doc_topics_mtx = lda.transform(corpus)

import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()

pyLDAvis.sklearn.prepare(lda, corpus, vectorizer)

