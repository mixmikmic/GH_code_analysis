from sml import execute

query = 'READ "../data/iris.csv" AND  SPLIT (train = .8, test = 0.2) AND  CLASSIFY (predictors = [1,2,3,4], label = 5, algorithm = svm)'

execute(query, verbose=True)

import pandas as pd
import numpy as np

from sklearn.preprocessing import label_binarize
import sklearn.cross_validation as cv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

names = ['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)', 'species']
data = pd.read_csv('../data/iris.csv', names=names)

iris_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
features = np.c_[data.drop('species',1).values]
labels = label_binarize(data['species'], classes=iris_classes)

n_classes = labels.shape[1]

(x_train, x_test, y_train, y_test) = cv.train_test_split(features, labels, test_size=0.25)

svm = OneVsRestClassifier(SVC(kernel='linear', probability=True))
model = svm.fit(x_train, y_train)
print('Accuracy:', model.score(x_test, y_test))



