from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# read in the iris data
iris = load_iris()
features = iris.data
labels = iris.target

# simulate splitting a dataset of 25 observations into 5 folds
from sklearn.cross_validation import KFold
kf = KFold(25, n_folds=5, shuffle=False)

# print the contents of each training and testing set
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], data[1]))


from sklearn.cross_validation import cross_val_score

# 10-fold cross-validation with K=5 for KNN
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, features, labels, cv=10, scoring='accuracy')
print(scores)

# use average accuracy as an estimate of out-of-sample accuracy
scores.mean()

# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, features, labels, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
    
#Return the index of the max score
print(k_scores.index(max(k_scores)))

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=13)
print(cross_val_score(knn, features, labels, cv=10, scoring='accuracy').mean())

# 10-fold cross-validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, features, labels, cv=10, scoring='accuracy').mean())

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
lm = LinearRegression()

labels = data['Sales']

# 10-fold cross-validation with all features
features = data[['TV', 'Radio', 'Newspaper']]
print(np.sqrt(-cross_val_score(lm, features, labels, cv=10, scoring='mean_squared_error')).mean())

# 10-fold cross-validation with two features (excluding Newspaper)
feature_cols = ['TV', 'Radio']
features = data[feature_cols]
print(np.sqrt(-cross_val_score(lm, features, labels, cv=10, scoring='mean_squared_error')).mean())



