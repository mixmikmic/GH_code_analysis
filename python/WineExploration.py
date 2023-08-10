import os
import sys
import logging
import warnings
import json
import copy
import re
import time
import itertools
from collections import defaultdict

import unidecode

import numpy as np

import pandas as pd
import pandas_profiling

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import linear_model

import gensim
import spacy
from spacy import displacy
from spacy.tokens import Doc
from spacy.lang.en import English

from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import pyLDAvis.gensim
pyLDAvis.enable_notebook()

# Uncomment to print Gensim log messages
# logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
# logging.root.level = logging.INFO

sns.set_style('darkgrid')
sns.set_context('notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
np.random.seed(42)

TYPES = {
    'country': str,
    'description': str,
    'designation': str,
    'points': int,
    'price': float,
    'province': str,
    'region_1': str,
    'region_2': str,
    'taster_name': str,
    'taster_twitter_handle': str,
    'title': str,
    'variety': str,
    'winery': str
}

df = pd.read_csv(
    './data/winemag-data-130k-v2.csv', dtype=TYPES, header=0, index_col=0)

with pd.option_context('display.max_rows', 3, 'display.max_columns', 300):
    display(df)

# pandas_profiling.ProfileReport(df, check_correlation=False, bins=10)

print('country: ', len(df['country'].unique()))
print('province: ', len(df['province'].unique()))
print('variety: ', len(df['variety'].unique()))

columns_to_merge = ['country', 'description', 'designation', 'province', 
                   'region_1', 'region_2', 'title', 'variety', 'winery']
df[columns_to_merge] = df[columns_to_merge].fillna('')

re_pattern = re.compile(r'-{1,2}[a-z]\.[a-z]\.?')  # Pattern to strip out name initials
def row2str(row):
    return re_pattern.sub(' ', unidecode.unidecode('. '.join(row[columns_to_merge])).lower())

start_time = time.time()
points, texts = zip(*[
    (row['points'], row2str(row))
    for _, row in df.iterrows()])
print('Converting dataframe took {} s'.format(time.time() - start_time))

print('points: ', len(points))
print('texts: ', len(texts))
print('Points:', points[0], ', text:', texts[0])

Doc.set_extension('filtered_tokens', default='')

def filter_token(t):
    return not (t.is_stop or t.is_punct or t.is_space)

def tokenize_component(doc):
    doc._.set('filtered_tokens', tuple(t.lemma_ for t in doc if filter_token(t)))
    return doc

nlp = English()
# nlp = spacy.load('en', disable=['parser', 'ner'])
nlp.add_pipe(tokenize_component)
print('NLP pipeline: ', nlp.pipe_names)

def gen_split_tokens(texts):
    return (doc._.filtered_tokens for doc in nlp.pipe(texts))

def gen_bigram(texts):
    texts = list(texts)
    bigram = gensim.models.phrases.Phrases(
        texts, 
        common_terms=set(["of", "with", "without", "and", "or", "the", "a"]))
    for t in texts:
        yield bigram[t]
        
def gen_tokens(texts):
    return gen_bigram(gen_split_tokens(t for t in texts))

get_ipython().run_line_magic('time', 'tokenized_texts = list(gen_tokens(texts))')
print('tokenized_texts: ', len(tokenized_texts))
for i in range(3):
    print('Example tokens: ', tokenized_texts[i][:10])

frequency = defaultdict(int)
for tokens in tokenized_texts:
    for token in tokens:
        frequency[token] += 1
        
wc = WordCloud(max_words=1000)
wc.fit_words(frequency)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

# Create a dictionary (This wil take a couple of minutes)
get_ipython().run_line_magic('time', 'id2word = gensim.corpora.Dictionary(tokenized_texts)')
print('id2word: ', id2word)

# ignore words that appear in less than 25 documents or more than 50% documents
id2word.filter_extremes(no_below=25, no_above=0.5)
id2word.compactify()
print('id2word: ', id2word, len(id2word))

get_ipython().run_line_magic('time', 'corpus = [id2word.doc2bow(tokens) for tokens in tokenized_texts]')

for k, v in itertools.islice(id2word.items(), 5):
    print('{}: {}'.format(k, v))

# TF-IDF transformed corpus
# TF-IDF will take a couple of seconds on the full corpus
get_ipython().run_line_magic('time', 'tfidf_model = gensim.models.TfidfModel(corpus, id2word=id2word)')

tf_idf_corpus = tfidf_model[corpus]
print('tf_idf_corpus: ', type(tf_idf_corpus), len(tf_idf_corpus))
get_ipython().run_line_magic('time', 'tfidf_csr = gensim.matutils.corpus2csc(tf_idf_corpus, num_terms=len(id2word), num_docs=len(tf_idf_corpus)).T')
print('tfidf_csr: ', type(tfidf_csr), tfidf_csr.shape)

# Print some of the TF-IDF transformations
for i in range(3):
    print('Example tokens: ', tokenized_texts[i][:2])
    bow_test = id2word.doc2bow(tokenized_texts[i])
    tfidf_sample = tfidf_model[bow_test]
    print(tfidf_sample[:2])
    print([tfidf_csr[i,tfidf_sample[x][0]] for x in range(2)])

x_train, x_test, y_train, y_test = train_test_split(
    tfidf_csr, np.array(points), test_size=0.05, random_state=42)

print('x_train: ', type(x_train), x_train.shape)
print('y_train: ', type(y_train), y_train.shape)
print('x_test: ', type(x_test), x_test.shape)
print('y_test: ', type(y_test), y_test.shape)

linreg_model = linear_model.Ridge(alpha=0.1)
get_ipython().run_line_magic('time', 'linreg_model.fit(x_train, y_train)')

# Predictions from train set
y_train_pred = linreg_model.predict(x_train)
print('y_train_pred: ', y_train_pred.shape)

print('Mean absolute error: {}'.format(mean_absolute_error(y_train, y_train_pred)))
print('Variance score: {}'.format(r2_score(y_train, y_train_pred)))

# Plot predictions
plt.figure(figsize=(10, 4))
plt.scatter(y_train, y_train_pred, alpha=.3)
plt.title('Linear regression TRAIN set')
plt.xlabel('Real values')
plt.ylabel('Predicted values')
plt.show()

# Predictions from test set
y_test_pred = linreg_model.predict(x_test)
print('y_test_pred: ', y_test_pred.shape)

print('Mean absolute error: {}'.format(mean_absolute_error(y_test, y_test_pred)))
print('Variance score: {}'.format(r2_score(y_test, y_test_pred)))

# Plot predictions
plt.figure(figsize=(10, 4))
plt.scatter(y_test, y_test_pred, alpha=.3)
plt.title('Linear regression TEST set')
plt.xlabel('Real values')
plt.ylabel('Predicted values')
plt.show()

# Find top features

# Top features contributing positive effect
max_indices = linreg_model.coef_.argsort()[-30:][::-1]
max_values = linreg_model.coef_[max_indices]

names = [id2word.get(i) for i in max_indices]
y_pos = np.arange(len(names))

fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(121)
ax1.barh(y_pos, max_values, align='center')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names)
ax1.invert_yaxis()  # labels read top-to-bottom
ax1.set_xlabel('Coefficient value')
ax1.set_title('Feature importance (top positive contribution)')


# Top features contributing negative effect 
min_indices = linreg_model.coef_.argsort()[:30]
min_values = linreg_model.coef_[min_indices]

names = [id2word.get(i) for i in min_indices]
y_pos = np.arange(len(names))

ax2 = plt.subplot(122)
ax2.barh(y_pos, min_values, align='center')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(names)
ax2.invert_yaxis()  # labels read top-to-bottom
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right')
ax2.set_xlabel('Coefficient value')
ax2.set_title('Feature importance (top negative contribution)')
plt.tight_layout()
plt.show()

### LDA topic-modelling on a subset of documents
num_lda_topics = 50
get_ipython().run_line_magic('time', 'lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_lda_topics, id2word=id2word, passes=2)')
# Serialise the LDA model
# lda_model.save('./data/lda_wiki.model')
# Serialise corpus transformed to LDA space
# %time gensim.corpora.MmCorpus.serialize('./data/wiki_lda.mm', lda_model[mm_corpus])
# lda_model.print_topics(10)

lda_topics = lda_model.get_topics()
print('lda_topics: ', type(lda_topics), lda_topics.shape)

lda_corpus = lda_model[corpus]
print('lda_corpus: ', type(lda_corpus), len(lda_corpus))
get_ipython().run_line_magic('time', 'lda_features_csr = gensim.matutils.corpus2csc(lda_corpus, num_terms=num_lda_topics, num_docs=len(lda_corpus)).T')
print('lda_features: ', type(lda_features_csr), lda_features_csr.shape)

pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

x_train_lda, x_test_lda, y_train_lda, y_test_lda = train_test_split(
    lda_features_csr, np.array(points), test_size=0.05, random_state=42)

print('x_train_lda: ', type(x_train_lda), x_train_lda.shape)
print('y_train_lda: ', type(y_train_lda), y_train_lda.shape)
print('x_test_lda: ', type(x_test_lda), x_test_lda.shape)
print('y_test_lda: ', type(y_test_lda), y_test_lda.shape)

linreg_lda_model = linear_model.Ridge(alpha=5)
get_ipython().run_line_magic('time', 'linreg_lda_model.fit(x_train_lda, y_train_lda)')

# Predictions from train set
y_train_lda_pred = linreg_lda_model.predict(x_train_lda)
print('y_train_lda_pred: ', y_train_lda_pred.shape)

print('Mean absolute error: {}'.format(mean_absolute_error(y_train_lda, y_train_lda_pred)))
print('Variance score: {}'.format(r2_score(y_train_lda, y_train_lda_pred)))

# Plot predictions
plt.figure(figsize=(10, 4))
plt.scatter(y_train_lda, y_train_lda_pred, alpha=.3)
plt.title('Linear regression LDA features TRAIN set')
plt.xlabel('Real values')
plt.ylabel('Predicted values')
plt.show()

# Predictions from test set
y_test_lda_pred = linreg_lda_model.predict(x_test_lda)
print('y_test_lda_pred: ', y_test_lda_pred.shape)

print('Mean absolute error: {}'.format(mean_absolute_error(y_test_lda, y_test_lda_pred)))
print('Variance score: {}'.format(r2_score(y_test_lda, y_test_lda_pred)))

# Plot predictions
plt.figure(figsize=(10, 4))
plt.scatter(y_test_lda, y_test_lda_pred, alpha=.3)
plt.title('Linear regression LDA features TEST set')
plt.xlabel('Real values')
plt.ylabel('Predicted values')
plt.show()

# Find top features

# Top postive contributing features
max_indices = linreg_lda_model.coef_.argsort()[-10:][::-1]
max_values = linreg_lda_model.coef_[max_indices]
y_pos = np.arange(len(max_indices))

fig = plt.figure(figsize=(10, 6))
ax1 = plt.subplot(121)
ax1.barh(y_pos, max_values, align='center')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(max_indices)
ax1.invert_yaxis()  # labels read top-to-bottom
ax1.set_xlabel('Coefficient value')
ax1.set_title('Feature importance (top positive contribution)')


# Top negative contributing features
min_indices = linreg_lda_model.coef_.argsort()[:10]
min_values = linreg_lda_model.coef_[min_indices]
y_pos = np.arange(len(min_indices))

ax2 = plt.subplot(122)
ax2.barh(y_pos, min_values, align='center')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(min_indices)
ax2.invert_yaxis()  # labels read top-to-bottom
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right')
ax2.set_xlabel('Coefficient value')
ax2.set_title('Feature importance (top negative contribution)')
plt.tight_layout()
plt.show()

def show_topic(idx):
    freqs = {}
    for widx in range(lda_topics.shape[1]):
        freqs[id2word[widx]] = lda_topics[idx, widx]

    wc = WordCloud(max_words=1000)
    wc.fit_words(freqs)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
print('GOOD:')
show_topic(max_indices[0])
show_topic(max_indices[1])

print('\n\n')
print('BAD:')
show_topic(min_indices[0])
show_topic(min_indices[1])

