from __future__ import print_function

import numpy as np

from sklearn import __version__ as sklearn_version
print('Sklearn version:', sklearn_version)

from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',
                 remove=('headers', 'footers', 'quotes'),
                 categories=categories, shuffle=True, random_state=42)

twenty_train.target_names

# Sample data
print(twenty_train.data[1])
print('---------------')
print('Target: ', twenty_train.target[0])

# Text preprocessing, tokenizing and filtering of stopwords

from sklearn.feature_extraction.text import CountVectorizer
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=5000,
                                stop_words='english')
X_train_counts = tf_vectorizer.fit_transform(twenty_train.data)
X_train_counts.shape

print(X_train_counts[0,:])
print(X_train_counts[:,0])

#From occurrences to frequencies
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tf = tfidf_transformer.transform(X_train_counts)
X_train_tf.shape

print(X_train_tf[0,:])
print(X_train_tf[:,0])

from sklearn.naive_bayes import MultinomialNB

# Define and fit in one line
clf = MultinomialNB().fit(X_train_tf, twenty_train.target)

#Score test data

# Read test data
twenty_test = fetch_20newsgroups(subset='test',
                 remove=('headers', 'footers', 'quotes'),
                 categories=categories, shuffle=True, random_state=42)

# Transform text to counts
X_test_counts = tf_vectorizer.transform(twenty_test.data)

# tf-idf transformation
X_test_tf = tfidf_transformer.transform(X_test_counts)

# Prediction
predicted = clf.predict(X_test_tf)

# Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy test: ', accuracy_score(twenty_test.target, predicted))

# Score 2 new docs
docs_new = ['God loves GPU', 'OpenGL on the GPU is fast']

def score_text(text_list):
    '''
    Score function
    '''
    X_new_counts = tf_vectorizer.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    return predicted


for doc, category in zip(docs_new, score_text(docs_new)):
    print('%r => %s' % (doc, twenty_train.target_names[category]))



