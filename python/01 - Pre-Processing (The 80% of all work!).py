from sklearn import datasets

datasets = datasets.load_iris()

print(datasets.data.shape)

print(datasets.target.shape)

datasets.data

datasets.target

X = datasets.data

y = datasets.target

X

y

import numpy as np
import urllib
import urllib.request

with urllib.request.urlopen("http://goo.gl/j0Rvxq") as url:
    s = url.read()

raw_data = s

dataset = np.loadtxt(raw_data, delimiter=",")

print(dataset.shape)

dataset[8]

y = dataset[:,8]

from sklearn import preprocessing

normalized_X = preprocessing.scale(X)

X

normalized_X

standarized_X = preprocessing.normalize(X)

X

standarized_X

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(datasets.data, datasets.target)

print(model.feature_importances_)

model.fit(X, y)

print(model.feature_importances_)





