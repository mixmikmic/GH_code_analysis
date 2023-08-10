import pandas as pd
from constants import *

census_data = pd.read_csv('combined_data.csv')

X = census_data[feature_cols]
y = census_data['Democrat']

from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# search for an optimal value of K for KNN
k_range = list(range(1, 41))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

from sklearn.grid_search import GridSearchCV

# define the parameter values that should be searched
k_range = list(range(1, 41))
leaf_size_range = list(range(20, 50))

param_grid = dict(n_neighbors=k_range, leaf_size=leaf_size_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]

print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

# 10-fold cross-validation with K=31 for KNN
knn = KNeighborsClassifier(n_neighbors=21, leaf_size=20)
scores = cross_val_score(knn, X, y, cv=10, scoring='roc_auc')

print(scores.mean())



