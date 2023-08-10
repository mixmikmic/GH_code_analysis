get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
# import cPickle -- only needed for saving production model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve

"""
If TextBlob doesn't import in a command line run:
pip install -U textblob
python -m textblob.download_corpora
"""

"""
Load, check data quantity, and look at a subset.

This file contains **a collection of more than 5 thousand SMS phone messages**
(see the `readme` file for more info).
"""
messages = [line.rstrip() for line in open('data/smsspamcollection/SMSSpamCollection')]
print('Number of messages loaded: ' + str(len(messages)))

# Print a sample of messages
for message_no, message in enumerate(messages[:10]):
    print(message_no, message)

# Let's get this file into a structured format
messages = pandas.read_csv('data/smsspamcollection/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["label", "message"])
messages.head(n=10)

# Let's take a look at the distribution of Spam and Ham
messages.groupby('label').describe()

# About 13% of the corpus is spam.

# Let's take a look the text message length
messages['length'] = messages['message'].map(lambda text: len(text))
messages.length.plot(figsize=(12,5), bins=50, kind='hist', label='Number of Characters',
                     title='Distribution of Message Lengths')
messages.length.describe()
messages.hist(figsize=(12,5), column='length', by='label', bins=50)

def split_into_tokens(message):
    return TextBlob(message).words

# Compare tokenized and non-tokenized
print('Raw Messages')
print(messages.message.head(n=5))
print('\nTokenized Messages - note the list object')
print(messages.message.head(n=5).apply(split_into_tokens))

# Use built in lemmatization

def split_into_lemmas(message):
    #message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

messages.message.head(n=5).apply(split_into_lemmas)

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
print(len(bow_transformer.vocabulary_))

def split_without_lemmas(message):
    #message = message.lower()
    #words = TextBlob(message).words
    return [word for word in message.split(' ')]

non_lemma_bow_transformer = CountVectorizer(analyzer=split_without_lemmas).fit(messages['message'])
print('Non lemmatized vocabular', len(non_lemma_bow_transformer.vocabulary_))

print(messages['message'][4])

# Let's run a sanity check 
message4 = messages['message'][4]
bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)

messages_bow = bow_transformer.transform(messages['message'])
print('sparse matrix shape:', messages_bow.shape)
print('number of non-zeros:', messages_bow.nnz)
print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['june']])

# Let's jump over to tfidf
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

get_ipython().magic("time spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])")

print('predicted:', spam_detector.predict(tfidf4)[0])
print('expected:', messages.label[4])

"""
Great! We fit the training data!
Let's see how well we fit the training data.
"""
all_predictions = spam_detector.predict(messages_tfidf)
print(all_predictions)
print('accuracy', accuracy_score(messages['label'], all_predictions))
print('confusion matrix\n', confusion_matrix(messages['label'], all_predictions))
print('(row=expected, col=predicted)')

plt.matshow(confusion_matrix(messages['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')

print(classification_report(messages['label'], all_predictions))

# TODO pull an example of when the training data is incorrectly fitted

msg_train, msg_test, label_train, label_test =     train_test_split(messages['message'], messages['label'], test_size=0.2)

print('Training set size: ', len(msg_train))
print('Testing set size: ', len(msg_test))
print('Total data set size: ', len(msg_train) + len(msg_test))

# Setup the pipeline for processing the data, transforming, and training
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

# Cross Validation in action!
scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=2,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=3,  # -1 = use all cores = faster
                         )
print('Raw Scores: ', scores)
print('Mean: ', scores.mean(), 'STD: ', scores.std())

# TODO Create your own modification

def custom_split_into_lemmas(message):
    message = message.lower() #uncomment this line for first level improvement
    words = TextBlob(message).words
    return [word.lemma for word in words]

messages.message.head(n=5).apply(custom_split_into_lemmas)

# Modify the pipeline to use the custom_split_into_lemmas
custom_pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=custom_split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])
custom_scores = cross_val_score(
    custom_pipeline,  # steps to convert raw messages into models
    msg_train,  # training data
    label_train,  # training labels
    cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
    scoring='accuracy',  # which scoring metric?
    n_jobs=-1,  # -1 = use all cores = faster
    )
print('Raw Scores: ', custom_scores)
print('Mean: ', custom_scores.mean(), 'STD: ', custom_scores.std())

