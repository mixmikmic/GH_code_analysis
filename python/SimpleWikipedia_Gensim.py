import os
import sys
import logging
import warnings
import copy
import re
import json
import tarfile
import itertools
import numpy as np

import gensim
# Uncomment to print Gensim log messages
# logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
# logging.root.level = logging.INFO

warnings.filterwarnings('ignore')
np.random.seed(42)

# Load data
from gensim.utils import smart_open
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki

DATA_PATH = './data/simplewiki-latest-pages-articles.xml.bz2'

def iter_wiki():
    for title, text, pageid in _extract_pages(smart_open(DATA_PATH)):
        text = filter_wiki(text)
        yield title, text

test_examples = [(ti, te) for ti, te in itertools.islice(iter_wiki(), 3)]
for title, text in test_examples:
    print('Title: ', title, ', text: len:', len(text), ', sample: ', repr(text[:50]))

# Tokenize
from gensim.parsing.preprocessing import STOPWORDS

def tokenize(text):
    return [token for token in gensim.utils.tokenize(text, lowercase=True, deacc=True) 
            if token not in STOPWORDS]

for title, text in test_examples:
    print('Tokens sample: ', tokenize(text)[:5])

# Create a dictionary (This wil take a couple of minutes)
doc_stream = (tokenize(text) for title, text in iter_wiki())
get_ipython().run_line_magic('time', 'id2word_wiki = gensim.corpora.Dictionary(doc_stream)')
print('id2word_wiki: ', id2word_wiki)

# ignore words that appear in less than 20 documents or more than 10% documents
id2word_wiki.filter_extremes(no_below=20, no_above=0.1)
print('id2word_wiki: ', id2word_wiki)

# Vectorize example
for title, text in test_examples:
    bow_test = id2word_wiki.doc2bow(tokenize(text))
    print(title, bow_test[:3], [id2word_wiki[i] for i, _ in bow_test[:3]])

# Example on how to find most common words
for title, text in test_examples:
    bow_test = id2word_wiki.doc2bow(tokenize(text))
    most_index, most_count = max(bow_test, key=lambda t: t[1])
    print(id2word_wiki[most_index], most_count)

# Serialise the corpus (takes a couple of minutes)
wiki_corpus_gen = (id2word_wiki.doc2bow(tokenize(text)) for title, text in iter_wiki())
get_ipython().run_line_magic('time', "gensim.corpora.MmCorpus.serialize('./data/wiki_bow.mm', wiki_corpus_gen)")

mm_corpus = gensim.corpora.MmCorpus('./data/wiki_bow.mm')
print('mm_corpus: ', mm_corpus)
print('mm_corpus[0]: ', mm_corpus[0][:7])

# LDA topic-modelling on a subset of documents
mm_corpus_subset = gensim.utils.ClippedCorpus(mm_corpus, 5000)
get_ipython().run_line_magic('time', 'lda_model = gensim.models.LdaModel(mm_corpus_subset, num_topics=10, id2word=id2word_wiki, passes=4)')
# Serialise the LDA model
# lda_model.save('./data/lda_wiki.model')
# Serialise corpus transformed to LDA space
# %time gensim.corpora.MmCorpus.serialize('./data/wiki_lda.mm', lda_model[mm_corpus])

# Print a the most imortant words for some of the topics
for i in range(lda_model.num_topics):
    topic = lda_model.print_topic(i, topn=7)
    print('Topic {}: '.format(i), topic)

for title, text in test_examples:
    bow_test = id2word_wiki.doc2bow(tokenize(text))
    print(title, lda_model[bow_test])

# TF-IDF transformed LSI topic modelling
# TF-IDF will take a couple of seconds on the full corpus
get_ipython().run_line_magic('time', 'tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=id2word_wiki)')
# Building LSI model on top of tf-idf will take a couple of minutes
get_ipython().run_line_magic('time', 'lsi_model = gensim.models.LsiModel(tfidf_model[mm_corpus], id2word=id2word_wiki, num_topics=200)')

# Print some of the TF-IDF transformations
for title, text in test_examples:
    bow_test = id2word_wiki.doc2bow(tokenize(text))
    print(title, tfidf_model[bow_test][:5])

print('Number of topics in LSI model: ', lsi_model.num_topics)
# Print a the most imortant words for some of the topics
for i in range(5):
    topic = lsi_model.print_topic(i, topn=7)
    print('Topic {}: '.format(i), topic)

# Print some of the LSI transformations
for title, text in test_examples:
    bow_test = id2word_wiki.doc2bow(tokenize(text))
    print(title, lsi_model[tfidf_model[bow_test]][:5])

# Test on unseen text
text = 'Physics is the study of energy, forces, mechanics, waves, and the structure of atoms and the physical universe.'

# Transform to BOW
bow_vector = id2word_wiki.doc2bow(tokenize(text))
print('bow_vector: ', [(id2word_wiki[id], count) for id, count in bow_vector])
print('')

# transform into LDA space
lda_vector = lda_model[bow_vector]
print('lda_vector: ', lda_vector)
print('Most important LDA topic: ', lda_model.print_topic(max(lda_vector, key=lambda item: item[1])[0]))
print('')

# Transform into the LSI space
lsi_vector = lsi_model[tfidf_model[bow_vector]]
print('Most important LSI topic: ', lsi_model.print_topic(max(lsi_vector, key=lambda item: abs(item[1]))[0]))


