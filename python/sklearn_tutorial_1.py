import pandas as pd

dataset = pd.read_csv('20news-18828.csv', header=None, delimiter=',', names=['label', 'text'])

print("There are 20 categories: %s" % (len(dataset.label.unique()) == 20))
print("There are 18828 records: %s" % (len(dataset) == 18828))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset.text, dataset.label, train_size=0.8)

from sklearn.feature_extraction.text import CountVectorizer
# initialize a CountVectorizer
cv = CountVectorizer()
# fit the raw data into the vectorizer and tranform it into a series of arrays
X_train_counts = cv.fit_transform(X_train)
X_train_counts.shape

X_test_counts = cv.transform(X_test)
X_test_counts.shape

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_counts, y_train)
predicted = clf.predict(X_test_counts)

# sample some of the predictions against the ground truths 
for prediction, truth in zip(predicted[:10], y_test[:10]):
    print(prediction, truth)

from sklearn import metrics

print(metrics.classification_report(y_test, predicted, labels=dataset.label.unique()))

p, r, f1, _ = metrics.precision_recall_fscore_support(y_test, predicted, labels=dataset.label.unique(), average='micro')

print("Micro-averaged Performance:\nPrecision: {0}, Recall: {1}, F1: {2}".format(p, r, f1))

print(metrics.confusion_matrix(y_test, predicted, labels=dataset.label.unique()))

import re
def normalize_numbers(s):
    return re.sub(r'\b\d+\b', 'NUM', s)

cv = CountVectorizer(preprocessor=normalize_numbers, ngram_range=(1,3), stop_words='english')

# fit the raw data into the vectorizer and tranform it into a series of arrays
X_train_counts = cv.fit_transform(X_train)
X_test_counts = cv.transform(X_test)

clf = MultinomialNB().fit(X_train_counts, y_train)
predicted = clf.predict(X_test_counts)
print(metrics.classification_report(y_test, predicted, labels=dataset.label.unique()))

from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(preprocessor=normalize_numbers, ngram_range=(1,3), stop_words='english')
X_train_tf = tv.fit_transform(X_train)
X_test_tf = tv.transform(X_test)
clf2 = MultinomialNB().fit(X_train_tf, y_train)
predicted = clf2.predict(X_test_tf)
print(metrics.classification_report(y_test, predicted, labels=dataset.label.unique()))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

from sklearn.linear_model import LogisticRegression

# for the sake of speed, we will just use all the default value of the constructor
cv = CountVectorizer()
X_train_counts = cv.fit_transform(X_train)
X_test_counts = cv.transform(X_test)

clf = LogisticRegression(solver='liblinear', max_iter=500, n_jobs=4)
clf.fit(X_train_counts, y_train)
predicted = clf.predict(X_test_counts)
print(metrics.classification_report(y_test, predicted, labels=dataset.label.unique()))

from sklearn.semi_supervised import LabelSpreading
from sklearn import preprocessing
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

label_prop_model = LabelSpreading()
# randomly select some of the training data and use them as unlabeled data
random_unlabeled_points = np.where(np.random.randint(0, 2, size=len(y_train)))
labels = np.copy(y_train)
# convert all labels into indecies, since we'll assign -1 to the unlabeled dataset
le = preprocessing.LabelEncoder()
le.fit(labels)
# transforms labels to numeric values
labels = le.transform(labels)
labels[random_unlabeled_points] = -1
labels_test = le.transform(y_test)

# here we will need to limit the features space, because we are going to convert the sparse
# matrices into dense ones required by the implementation of LabelSpreading. 
# Using dense matrices in general slows down training and takes up a lot of memory; therefore,
# in most cases, it is not recommended.
cv = CountVectorizer(max_features=100)
X_train_counts = cv.fit_transform(X_train)
X_test_counts = cv.transform(X_test)
# toarray() here is to convert a sparse matrix into a dense matrix, as the latter is required
# by the algorithm. 
label_prop_model.fit(X_train_counts.toarray(), labels)
predicted = label_prop_model.predict(X_test_counts.toarray())

# For this experiment, we are only using 100 most frequent words as features in the train set, 
# so the test results will be lousy. 
p, r, f1, _ = metrics.precision_recall_fscore_support(labels_test, predicted, average='macro')
print("Macro-averaged Performance:\nPrecision: {0}, Recall: {1}, F1: {2}".format(p, r, f1))

from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn import preprocessing

X_train, X_test, y_train, y_test = train_test_split(dataset.text, dataset.label, train_size=0.8)
# I reduced the number of features here just to make training a little faster
cv = CountVectorizer(max_features=200)
X_train_counts = cv.fit_transform(X_train)
X_test_counts = cv.transform(X_test)

kmeans = KMeans(n_clusters=20, random_state=0)
kmeans.fit(X_train_counts)
# see the cluster label for each instance
print(kmeans.labels_)
print(kmeans.predict(X_test_counts))

le = preprocessing.LabelEncoder()
le.fit(y_train)
# you can also pass in the labels, if you have some annotated data
kmeans.fit(X_train_counts, le.transform(y_train))
predicted = kmeans.predict(X_test_counts)
p, r, f1, _ = metrics.precision_recall_fscore_support(le.transform(y_test), predicted, average='macro')
print("Macro-averaged Performance:\nPrecision: {0}, Recall: {1}, F1: {2}".format(p, r, f1))

from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv('20news-18828.csv', header=None, delimiter=',', names=['label', 'text'])
X = dataset.text
y = dataset.label

# the train_test_split() function shuffles the dataset under the hood, but the KFold object
# does not; therefore, if your dataset is sorted, make sure to shuffle it. For the sake of time,
# we are doing 5 fold 
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    cv = CountVectorizer()
    X_train_counts = cv.fit_transform(X_train)
    X_test_counts = cv.transform(X_test)
    clf = MultinomialNB().fit(X_train_counts, y_train)
    predicted = clf.predict(X_test_counts)
    p, r, f1, _ = metrics.precision_recall_fscore_support(y_test, predicted, average='macro')
    print("Macro-averaged Performance:\nPrecision: {0}, Recall: {1}, F1: {2}".format(p, r, f1))

