from sklearn import datasets
import numpy as np

digits = datasets.load_digits()
print(digits['DESCR'])
print("Data shape: {}".format(digits.data.shape))
print("Target shape: {}".format(digits.target.shape))
np.column_stack([digits.data, digits.target])

import matplotlib.pyplot as plt

def display_digit(digit_data):
    div = np.full((8,8), 16, dtype=int)
    dig = digit_data.reshape(8,8) / div
    plt.imshow(dig, cmap="gray")
    plt.show()

sample = 1796 # change this to anything between 0 and size of data set (1796) to view a plot of any data point
print("Sample label: {}".format(digits.target[sample]))
display_digit(digits.data[sample])

from sklearn.svm import LinearSVC

# Step 1: Create the classifier, specifying parameters (or use defaults)
clf = LinearSVC()

# Step 2: Fit the data!
clf.fit(digits.data, digits.target)

# Step 3: Try predicting something
pred_num = 1796 # Change this to anything between 0 and the size of data set (1796) to predict a different data point
pred = clf.predict([digits.data[pred_num]])
print("Classifier predicted {}".format(pred))
print("Actual value: {}".format(digits.target[pred_num]))
display_digit(digits.data[pred_num])

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# predict everything we originally trained on
pred = clf.predict(digits.data)
print("Accuracy: {}".format(accuracy_score(digits.target, pred)))
print(classification_report(digits.target, pred))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.33, random_state=1)

print("Train X shape: {}".format(X_train.shape))
print("Test X shape: {}".format(X_test.shape))

from sklearn.neighbors import KNeighborsClassifier

# If there are n features in the data and we were to plot the points in an n dimensional space
# KNN will assign a label based on the labels of the closest k points to the data point we wish to predict
clf = KNeighborsClassifier(n_neighbors=5) # Here we let k=5
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train) # predict using the training data
accuracy_score(y_train, y_pred)

clf = KNeighborsClassifier(n_neighbors=1) # now k=1
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train) # predict using the training data
accuracy_score(y_train, y_pred)

y_pred = clf.predict(X_test) # predict using the training data
accuracy_score(y_test, y_pred)

k_results = []
for i in range(1, 20):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    k_results.append(accuracy_score(y_test, y_pred))

plt.plot(k_results)
plt.show()

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

print("FileNames: {}".format(newsgroups_train.filenames.shape));
print("Target: {}".format(newsgroups_train.target.shape));
datapoint=1 # view a data point, pick from 0 to 11313
print(newsgroups_train.data[datapoint])
print(newsgroups_train.target_names[newsgroups_train.target[datapoint]])

from sklearn.feature_extraction.text import CountVectorizer

# Create bag-of-words representation of text, ignoring stopwords (like "the" or "of", etc.)
vectorizer = CountVectorizer(stop_words='english') 
vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_train.shape

vocab = vectorizer.vocabulary_
vocab

# Let's view a vector for the datapoint we saw earlier
# vectors is a sparse matrix, so we have to convert to a dense matrix.
data_mat = vectors_train[datapoint].todense()
print(data_mat.shape)
data_mat

# let's see the count for a specific word
data_mat[(0,vocab['clock'])]

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(vectors_train, newsgroups_train.target)

# Get test data, we must use the same vectorizer or else we'll end up with a different feature set!
newsgroups_test = fetch_20newsgroups(subset='test')
vectors_test = vectorizer.transform(newsgroups_test.data)
vectors_test.shape

y_pred = clf.predict(vectors_test)
y_pred.shape
accuracy_score(newsgroups_test.target, y_pred)

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile

ch2 = SelectPercentile(chi2, percentile=5) # use "percentile" best features
y_train = newsgroups_train.target
X_train = ch2.fit_transform(vectors_train, y_train)
y_test = newsgroups_test.target
X_test = ch2.transform(vectors_test)

inv_vocab = {v: k for k, v in vocab.items()} # maps from index to word

# list most important words
feature_names = [inv_vocab[i] for i in ch2.get_support(indices=True)]
feature_names

# Train with new set
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred.shape
accuracy_score(y_test, y_pred)

category = 1
print(newsgroups_train.target_names[category])
feature_coefs = np.column_stack([np.array(feature_names), clf.coef_[category]])
feature_coefs = np.core.records.fromarrays(feature_coefs.transpose(), names='feature, coef', formats = 'S8, f8')
feature_coefs = np.sort(feature_coefs, order=['coef'], kind='mergesort')
feature_coefs.shape

[x[0] for x in feature_coefs[-10:-1]] # print 10 best features

