get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

from figures import plot_tree_interactive
plot_tree_interactive()

from figures import plot_forest_interactive
plot_forest_interactive()

from sklearn import grid_search
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
parameters = {'max_features':['sqrt', 'log2'],
              'max_depth':[5, 7, 9]}

clf_grid = grid_search.GridSearchCV(rf, parameters)
clf_grid.fit(X_train, y_train)

clf_grid.score(X_train, y_train)

clf_grid.score(X_test, y_test)



