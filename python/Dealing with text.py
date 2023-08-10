# Import the libraries we will be using
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = 10, 8

np.random.seed(36)

# We will want to keep track of some different roc curves, lets do that here
tprs = []
fprs = []
roc_labels = []

get_ipython().system('head -2 data/spam_ham.csv')

get_ipython().system("cut -f2 -d',' data/spam_ham.csv | sort | uniq -c | head")

data = pd.read_csv("data/spam_ham.csv", quotechar="'", escapechar="\\")

data.head()

data['spam'] = pd.Series(data['spam'] == 'spam', dtype=int)

data.head()

X = data['text']
Y = data['spam']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.75)

binary_vectorizer = CountVectorizer(binary=True)
binary_vectorizer.fit(X_train)

binary_vectorizer.vocabulary_.keys()[0:10]

X_train_binary = binary_vectorizer.transform(X_train)
X_test_binary = binary_vectorizer.transform(X_test)

X_test_binary

X_test_binary[0:20, 0:20].todense()

model = LogisticRegression()
model.fit(X_train_binary, Y_train)

print "Area under the ROC curve on the test data = %.3f" % metrics.roc_auc_score(model.predict(X_test_binary), Y_test)

fpr, tpr, thresholds = metrics.roc_curve(Y_test, model.predict_proba(X_test_binary)[:,1])
tprs.append(tpr)
fprs.append(fpr)
roc_labels.append("Default Binary")
ax = plt.subplot()
plt.plot(fpr, tpr)
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC Curve")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()

# Fit a counter
count_vectorizer = CountVectorizer()
count_vectorizer.fit(X_train)

# Transform to counter
X_train_counts = count_vectorizer.transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_counts, Y_train)

print "Area under the ROC curve on the test data = %.3f" % metrics.roc_auc_score(model.predict(X_test_counts), Y_test)

fpr, tpr, thresholds = metrics.roc_curve(Y_test, model.predict_proba(X_test_counts)[:,1])
tprs.append(tpr)
fprs.append(fpr)
roc_labels.append("Default Counts")
ax = plt.subplot()
plt.plot(fpr, tpr)
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC Curve")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()

# Fit a counter
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_train)

# Transform to a counter
X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_tfidf, Y_train)

print "Area under the ROC curve on the test data = %.3f" % metrics.roc_auc_score(model.predict(X_test_counts), Y_test)

fpr, tpr, thresholds = metrics.roc_curve(Y_test, model.predict_proba(X_test_tfidf)[:,1])
tprs.append(tpr)
fprs.append(fpr)
roc_labels.append("Default Tfidf")
ax = plt.subplot()
plt.plot(fpr, tpr)
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC Curve")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()

for fpr, tpr, roc_label in zip(fprs, tprs, roc_labels):
    plt.plot(fpr, tpr, label=roc_label)

plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC Curves")
plt.legend()
plt.xlim([0, .07])
plt.ylim([.98, 1])
plt.show()

model = BernoulliNB()
model.fit(X_train_tfidf, Y_train)

print "AUC on the count data = %.3f" % metrics.roc_auc_score(model.predict(X_test_tfidf), Y_test)

