from os.path import join

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import gensim
import numpy as np
import scipy as sp

from sklearn.datasets import fetch_20newsgroups


get_ipython().run_line_magic('run', '../src/load_data_utils.py')
get_ipython().run_line_magic('run', '../src/glove_2_word2vec.py')

__, DATA_DIR = get_env_vars(True)
VECTORS_DIR = join(DATA_DIR, 'glove.6B')
FILENAME_PREFIX = 'glove.6B.100d'
VECTORS_FILENAME_DIR = join(VECTORS_DIR, FILENAME_PREFIX + '.txt')

newsgroups_train = fetch_20newsgroups()
newsgroups_test = fetch_20newsgroups(subset='test')

model = gensim.models.KeyedVectors.load(str(join(VECTORS_DIR, FILENAME_PREFIX + '.w2v')))

model.most_similar(positive=['king', 'woman'], negative=['man'])

from collections import OrderedDict
import re

TOKENIZING_PATTERN = '(?u)\\b\\w\\w+\\b'

def preprocess_texts(sentences_list):
  return (list(
    map(
      lambda sentence: 
        ' '.join(OrderedDict.fromkeys(re.findall(TOKENIZING_PATTERN, sentence)))
          .lower(),
      sentences_list)))

X_train_text = preprocess_texts(newsgroups_train['data'])
X_test_text = preprocess_texts(newsgroups_test['data'])

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(token_pattern=TOKENIZING_PATTERN, min_df=2, max_df=0.025)

X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)

X_train.shape

y_train = newsgroups_train['target']
y_test = newsgroups_test['target']

from sklearn.decomposition import TruncatedSVD

tsvd = TruncatedSVD(n_iter=50)
get_ipython().run_line_magic('time', 'X_reduced = tsvd.fit_transform(X_train)')

plt.figure(figsize=(16, 12))

plt.scatter(*X_reduced.T, c=y_train)
plt.show()

def encode_sentence(glove_model, sent, weights=None):
  if weights is None:
    normalizing_factor = len(sent)
    word_vectors = (glove_model[w] for w in sent if glove_model.vocab.get(w))
  else:
    normalizing_factor = len([w for w in sent if glove_model.vocab.get(w)])
    word_vectors = (glove_model[w] * weights[i] for (i, w) in enumerate(sent) if glove_model.vocab.get(w))
  return sum(word_vectors) / normalizing_factor

get_ipython().run_line_magic('time', 'X_glove_train = np.array([encode_sentence(model, s.split()) for s in X_train_text])')
get_ipython().run_line_magic('time', 'X_glove_test = np.array([encode_sentence(model, s.split()) for s in X_test_text])')

from operator import itemgetter
from itertools import groupby

def encode_sentences(X_tfidf, tfidf):
  def get_nonzeros(v):
    try:
      nz = v[v.nonzero()].tolist()
    except AttributeError:
      print(v[v.nonzero()])
    else:
      return v[v.nonzero()].tolist()[0]
  retrieved_sentences = tfidf.inverse_transform(X_tfidf)
  return (np.array(
    [
      encode_sentence(
        model, 
        retrieved_sentences[i],
        weights=get_nonzeros(X_tfidf[i, :]))
      for i in range(X_tfidf.shape[0])]))

get_ipython().run_line_magic('time', 'X_glove_weighted_train = encode_sentences(X_train, tfidf)')
get_ipython().run_line_magic('time', 'X_glove_weighted_test = encode_sentences(X_test, tfidf)')

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

get_ipython().run_line_magic('time', 'X_glove_weighted_train_pca = pca.fit_transform(X_glove_weighted_train)')
get_ipython().run_line_magic('time', 'X_glove_train_pca = pca.fit_transform(X_glove_train)')

plt.figure(figsize=(16, 12))

plt.title('Mean averaged word embeddings')
plt.scatter(*X_glove_train_pca.T, c=y_train)
plt.show()

plt.figure(figsize=(16, 12))

plt.title('Tfidf weight-averaged word embeddings')
plt.scatter(*X_glove_weighted_train_pca.T, c=y_train)
plt.show()

from sklearn.metrics import f1_score 
from sklearn.linear_model import SGDClassifier 

sparse_lreg = SGDClassifier(n_iter=25, alpha=0.0001)

get_ipython().run_line_magic('time', 'sparse_lreg.fit(X_train, y_train)')

print('accuracy:', round(sparse_lreg.score(X_test, y_test), 3))
print('f1:', round(f1_score(y_test, sparse_lreg.predict(X_test), average='weighted'), 3))

from sklearn.preprocessing import StandardScaler

dense_lreg = SGDClassifier(n_iter=50, alpha=0.00005)

sscaler = StandardScaler()

X_glove_normalized_train = sscaler.fit_transform(X_glove_train)

get_ipython().run_line_magic('time', 'dense_lreg.fit(X_glove_train, y_train)#, classes=np.unique(y_train))')

print('accuracy:', round(dense_lreg.score(X_glove_test, y_test), 3))
print('f1:', round(f1_score(y_test, dense_lreg.predict(X_glove_test), average='weighted'), 3))

dense_lreg = SGDClassifier(n_iter=50, alpha=0.00001)

get_ipython().run_line_magic('time', 'dense_lreg.fit(X_glove_weighted_train, y_train)')

print('accuracy:', round(dense_lreg.score(X_glove_weighted_test, y_test), 3))
print('f1:', round(f1_score(y_test, dense_lreg.predict(X_glove_weighted_test), average='weighted'), 3))

n = 5
no_closest=5

text_encodings = [X_glove_train[i, :] for i in range(n)]
text_encodings_weighted = [X_glove_weighted_train[i, :] for i in range(n)]

model.most_similar([text_encodings[0]], topn=5)

model.most_similar([text_encodings[1]], topn=5)

model.most_similar([text_encodings[2]], topn=5)

model.most_similar([text_encodings_weighted[0]], topn=5)

model.most_similar([text_encodings_weighted[1]], topn=5)

model.most_similar([text_encodings_weighted[2]], topn=5)

X_train_text[0]

[(w, w in model.vocab) for w in tfidf.inverse_transform(X_train[0, :])[0]]

