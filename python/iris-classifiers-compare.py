get_ipython().magic('matplotlib inline')
from sklearn import datasets

from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris = datasets.load_iris()

df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])

df.head()

df.tail()

df.describe()

X = iris.data[0:150, :]
X.shape

Y = iris.target[0:150]
Y.shape

model_tree = tree.DecisionTreeClassifier()
model_svm = SVC()
model_per = Perceptron()
model_sgd = SGDClassifier()
model_KNN = KNeighborsClassifier()
model_GNB = GaussianNB()

model_tree.fit(X, Y)
model_svm.fit(X, Y)
model_per.fit(X, Y)
model_sgd.fit(X, Y)
model_KNN.fit(X, Y)
model_GNB.fit(X, Y)


# Testing using the same data
pred_tree = model_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {0:.{1}f}'.format(acc_tree, 0))

pred_svm = model_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {0:.{1}f}'.format(acc_svm, 0))

pred_per = model_per.predict(X)
acc_per = accuracy_score(Y, pred_per) * 100
print('Accuracy for Perceptron: {0:.{1}f}'.format(acc_per, 0))

pred_sgd = model_sgd.predict(X)
acc_sgd = accuracy_score(Y, pred_sgd) * 100
print('Accuracy for SGD: {0:.{1}f}'.format(acc_sgd, 0))

pred_KNN = model_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {0:.{1}f}'.format(acc_KNN, 0))

pred_GNB = model_GNB.predict(X)
acc_GNB = accuracy_score(Y, pred_GNB) * 100
print('Accuracy for GaussianNB: {0:.{1}f}'.format(acc_GNB, 0))


# The best classifier
best = np.argmax([acc_svm, acc_per, acc_KNN, acc_tree, acc_GNB, acc_sgd])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN', 3: 'DecisionTree', 4: 'GNB', 5: 'SGD'}
print('Best iris classifier is {}'.format(classifiers[best]))

