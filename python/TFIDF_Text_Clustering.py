get_ipython().magic('pprint')

get_ipython().magic('matplotlib inline')

# Standard imports
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pprint
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

import string
import nltk
from nltk.stem.porter import PorterStemmer


text = 'Information retrieval is different from Data mining'

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(1,3))

tk_func = cv.build_analyzer()

import pprint
pp = pprint.PrettyPrinter(indent=2, depth=1, width=80, compact=True)

pp.pprint(tk_func(text))

in_list = []
in_list.append(text)

in_list

cv = cv.fit(in_list)

cv

cv.vocabulary_.items()

import operator
my_voc = sorted(cv.vocabulary_.items(), key=operator.itemgetter(1))
my_voc

print('Token mapping:')
print(40*'-')

for tokens, rank in my_voc:
    print(rank, tokens)

import nltk
mvr = nltk.corpus.movie_reviews

from sklearn.datasets import load_files

data_dir = '/usr/share/nltk_data/corpora/movie_reviews'

mvr = load_files(data_dir, shuffle = False)
print('Number of Reviews: {0}'.format(len(mvr.data)))

from sklearn.model_selection import train_test_split

mvr_train, mvr_test, y_train, y_test = train_test_split(
    mvr.data, mvr.target, test_size=0.25, random_state=23)

print('Number of Reviews in train: {0}'.format(len(mvr_train)))
print('Number of Reviews in test: {0}'.format(len(mvr_test)))

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

tools = [('cv', CountVectorizer()), ('nb', MultinomialNB())]
pclf = Pipeline(tools)

# Lowercase and restrict ourselves to about half the available features
pclf.set_params(cv__stop_words = 'english',                 cv__ngram_range=(1,2),                 cv__lowercase=True)

pclf

pclf.fit(mvr_train, y_train)
y_pred = pclf.predict(mvr_test)
print(metrics.classification_report(y_test, y_pred, target_names = mvr.target_names))

# Extract the classifier
clf = pclf.steps[1][1]
print('Number of Features = {}'.format(clf.feature_log_prob_.shape[1]))

pclf.set_params(cv__stop_words = 'english',                 cv__ngram_range=(1,3),                 cv__lowercase=True,                 cv__min_df=2,                 cv__max_df=0.5)

pclf.fit(mvr_train, y_train)
y_pred = pclf.predict(mvr_test)
print(metrics.classification_report(y_test, y_pred, target_names = mvr.target_names))

pclf.steps

pclf.steps[1]

pclf.steps[1][1].feature_log_prob_

# Extract the classifier
clf = pclf.steps[1][1]
print('Number of Features = {}'.format(clf.feature_log_prob_.shape[1]))

from sklearn.cluster import KMeans

true_k = 2

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

from sklearn.feature_extraction.text import CountVectorizer

# Verify attributes

cv = CountVectorizer(stop_words = 'english',                      ngram_range=(1, 3), max_features=100000)

train_counts = cv.fit_transform(mvr_train)
test_data = cv.fit_transform(mvr_test)

km.fit(test_data)

top_tokens = 20
labels = ['Neg', 'Pos']

print('Top {} tokens per cluster:\n'.format(top_tokens))

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = cv.get_feature_names()

for idx in range(true_k):
    print("Cluster {0}:".format(idx), end='')
    
    for jdx in order_centroids[idx, :top_tokens]:
        print(' {0}'.format(terms[jdx]), end='')
        
    print('\n')

# Split into training and testing
from sklearn.datasets import fetch_20newsgroups

train = fetch_20newsgroups(data_home='/dsa/data/DSA-8630/newsgroups/',                            subset='train', shuffle=True, random_state=23)

test = fetch_20newsgroups(data_home='/dsa/data/DSA-8630/newsgroups/',                           subset='test', shuffle=True, random_state=23)

# Following Example was inspired by scikit learn demo
# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

# First, train on normal set of features, baseline performance.

train_counts = tf.fit_transform(train['data'])
test_data = tf.transform(test['data'])

nb = MultinomialNB()
nb = nb.fit(train_counts, train['target'])
predicted = nb.predict(test_data)

print("Prediction accuracy = {0:5.1f}%".format(100.0 * nb.score(test_data, test['target'])))
print('Number of Features = {}'.format(nb.feature_log_prob_.shape[1]))

# Now employ feature selection
from sklearn.feature_selection import SelectKBest, chi2

# Number of features to keep
num_k = 10000

# Employ select k best wiht chi2 since this works with sparse matrices.
ch2 = SelectKBest(chi2, k=num_k)
xtr = ch2.fit_transform(train_counts, train['target'])
xt = ch2.transform(test_data)

# Train simple model and compute accuracy
nb = nb.fit(xtr, train['target'])
predicted = nb.predict(xt)

print("NB prediction accuracy = {0:5.1f}%".format(100.0 * nb.score(xt, test['target'])))
print('Number of Features = {}'.format(nb.feature_log_prob_.shape[1]))

feature_names = tf.get_feature_names()

indices = ch2.get_support(indices=True)
feature_names = np.array([feature_names[idx] for idx in indices])

import pprint
pp = pprint.PrettyPrinter(indent=2, depth=1, width=80, compact=True)

top_count = 5

for idx, target in enumerate(train['target_names']):
    top_names = np.argsort(nb.coef_[idx])[-top_count:]
    tn_lst = [name for name in feature_names[top_names]]
    tn_lst.reverse()

    print('\n{0}:'.format(target))
    pp.pprint(tn_lst)

