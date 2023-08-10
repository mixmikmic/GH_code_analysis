from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

# The digits dataset
print(digits.data)

# The iris dataset
print(iris.data)

# The digits dataset class attribute possible values (our classification labels)
digits.target

# Shape of the digits dataset (each original sample is an image of shape (8, 8))
digits.images[0]

# Implement the support vector machine estimator
from sklearn import svm
# clf == classifier
clf = svm.SVC(gamma=0.001, C=100.)

# Fit classifier to the model by passing our training set to the fit method
# As a training set we'll use all the images of our dataset apart from the last one
clf.fit(digits.data[:-1], digits.target[:-1])

# Now we predict the final digit image, based on our learned classifier
clf.predict(digits.data[-1:])

# Build a classifier

from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

# Pickle it

import pickle

s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])

# Unless otherwise specified, input will be cast to float64

import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
X.dtype

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
X_new.dtype

# Regression targets are cast to float64, classification targets are maintained

from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
clf = SVC()
clf.fit(iris.data, iris.target)

list(clf.predict(iris.data[:3]))

clf.fit(iris.data, iris.target_names[iris.target])

list(clf.predict(iris.data[:3]))

