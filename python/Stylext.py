# These are the core libraries you need to import to run the scripts that follow.

import pandas as pd
import numpy as np
import scipy as sp

# Here are more specific tools from Scikit-Learn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # two vectorization methods we want for later
from sklearn.naive_bayes import MultinomialNB # multinomial naive bayes classifier
from sklearn.linear_model import LogisticRegression # basic logistic regression classifier
from sklearn.cross_validation import train_test_split # this splits the data loaded in into training & testing groups
from sklearn import metrics # this will help us understand the results of the train/test split simulation

# Read post_feed.csv into a DataFrame. Any CSV with columns containing raw tweet contents and usernames can often work.
# If you're offline, replace the link with the file location for post_feed.csv if you have it stored locally.

url = 'https://raw.githubusercontent.com/analyticascent/stylext/master/csv/post_feed.csv'
post = pd.read_csv(url)


# define X and y, or the manipulated variable and the responding variable: Given the text, which user tweeted it?

X = post.raw_text  # Depending on the raw tweet text column contents...
y = post.username  # ...which user wrote the tweet?


# split the new DataFrame into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# check the first five rows/tweets

post.head()

# check the first five rows in a shorter format

X.head()

# check the number of rows and columns

X.shape

# use CountVectorizer to create document-term matrices from X_train and X_test

vect = CountVectorizer() # because vect is way easier to type than CountVectorizer...
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# now we have quantitative info about the tweets that a 'multinomial naive Bayes classifier' can work with

vect

# rows are documents, columns are terms (aka "tokens" or "features")

X_train_dtm.shape

# last 50 features

print vect.get_feature_names()[-50:]

# show vectorizer options

vect

# We will not convert to lowercase for now, but if we did it would reduce the number of quantified features

vect = CountVectorizer(lowercase=False)
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape

# last 50 features

print vect.get_feature_names()[-50:]

# include 1-grams and 2-grams

vect = CountVectorizer(ngram_range=(1, 2))
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape

# last 50 features

print vect.get_feature_names()[-50:]

# use default options for CountVectorizer
vect = CountVectorizer()

# create document-term matrices
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# use Naive Bayes to predict the star rating
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy
print metrics.accuracy_score(y_test, y_pred_class)

# define a function that accepts a vectorizer and calculates the accuracy

def tokenize_test(vect):
    X_train_dtm = vect.fit_transform(X_train)
    print 'Features: ', X_train_dtm.shape[1]
    X_test_dtm = vect.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    print 'Accuracy: ', metrics.accuracy_score(y_test, y_pred_class)

vect = CountVectorizer()
tokenize_test(vect)

# include 1-grams and 2-grams

vect = CountVectorizer(ngram_range=(1, 2))
tokenize_test(vect)

# show vectorizer options

vect

# remove English stop words

vect = CountVectorizer(stop_words='english')
tokenize_test(vect)

# set of stop words

print vect.get_stop_words()

# remove English stop words and only keep 100 features

vect = CountVectorizer(stop_words='english', max_features=100)
tokenize_test(vect)

# all 100 features

print vect.get_feature_names()

# include 1-grams and 2-grams, and limit the number of features

vect = CountVectorizer(ngram_range=(1, 2), max_features=2200)
tokenize_test(vect)

# include 1-grams and 2-grams

vect = CountVectorizer(ngram_range=(1, 2))
tokenize_test(vect)

# include 1-grams and 2-grams, and limit the number of features

vect = CountVectorizer(ngram_range=(1, 2), max_features=10000)
tokenize_test(vect)

# include 1-grams and 2-grams, and only include terms that appear at least 2 times

vect = CountVectorizer(ngram_range=(1, 2),  max_features=10000, min_df=2)
tokenize_test(vect)
print vect.get_feature_names()

# Just pretend each of these strings is a "document" - we will vectorize them

simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']

# Term Frequency

vect = CountVectorizer()
tf = pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())
tf

# Document Frequency

vect = CountVectorizer(binary=True)
df = vect.fit_transform(simple_train).toarray().sum(axis=0)
pd.DataFrame(df.reshape(1, 6), columns=vect.get_feature_names())

# Term Frequency-Inverse Document Frequency (simple version)

tf/df

# TfidfVectorizer

vect = TfidfVectorizer()
pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())

# Term Frequency

vect = CountVectorizer()
tf = pd.DataFrame(vect.fit_transform(post).toarray(), columns=vect.get_feature_names())
tf

# Document Frequency

vect = CountVectorizer(binary=True)
df = vect.fit_transform(simple_train).toarray().sum(axis=0)
pd.DataFrame(df.reshape(1, 6), columns=vect.get_feature_names())

# Term Frequency-Inverse Document Frequency (simple version)

tf/df

# TfidfVectorizer

vect = TfidfVectorizer()
pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())

# define X and y

feature_cols = ['raw_text', 'syllables', 'periods', 'hyphens']
X = post[feature_cols]
y = post.username

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# use CountVectorizer with text column only

vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train.raw_text)
X_test_dtm = vect.transform(X_test.raw_text)
print X_train_dtm.shape
print X_test_dtm.shape

# shape of other four feature columns

X_train.drop('raw_text', axis=1).shape

# cast other feature columns to float and convert to a sparse matrix

extra = sp.sparse.csr_matrix(X_train.drop('raw_text', axis=1).astype(float))
extra.shape

# combine sparse matrices

X_train_dtm_extra = sp.sparse.hstack((X_train_dtm, extra))
X_train_dtm_extra.shape

# repeat for testing set

extra = sp.sparse.csr_matrix(X_test.drop('raw_text', axis=1).astype(float))
X_test_dtm_extra = sp.sparse.hstack((X_test_dtm, extra))
X_test_dtm_extra.shape

# use logistic regression with text column only

logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm, y_train)
y_pred_class = logreg.predict(X_test_dtm)
print metrics.accuracy_score(y_test, y_pred_class)

# use logistic regression with all features

logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm_extra, y_train)
y_pred_class = logreg.predict(X_test_dtm_extra)
print metrics.accuracy_score(y_test, y_pred_class)



