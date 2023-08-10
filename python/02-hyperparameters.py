get_ipython().magic("config InlineBackend.figure_format='retina'")
get_ipython().magic('matplotlib inline')

import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["font.size"] = 14

# Let's use our trusty blob dataset again
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=2000, centers=20, random_state=42)
labels = ["b", "r"]
y = np.take(labels, (y < 10))

plt.figure()
for label in labels:
    mask = (y == label)
    plt.scatter(X[mask, 0], X[mask, 1], c=label, s=40)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()

from sklearn.ensemble import RandomForestClassifier

RandomForestClassifier().get_params()

from sklearn.cross_validation import cross_val_score
import scipy

clf = RandomForestClassifier()
# should you always use accuracy??
scores = cross_val_score(clf, X, y, scoring='accuracy')
print('scores:', scores)
print('mean:', np.mean(scores), "SEM:", scipy.stats.sem(scores))

from sklearn.grid_search import GridSearchCV

parameter_grid = {'max_depth': [1, 2, 4, 8, 16, 32, 64],
                  'n_estimators': [10, 20, 40, 80, 120, 160]}
grid_search = GridSearchCV(RandomForestClassifier(), parameter_grid)

grid_search.fit(X, y)

grid_search.grid_scores_

def plot_scores(grid):
    scores = [config.mean_validation_score for config in grid]
    max_depth = [config.parameters['max_depth'] for config in grid]
    n_estimators = [config.parameters['n_estimators'] for config in grid]

    plt.scatter(max_depth, n_estimators, c=scores, s=60, lw=0, cmap='viridis_r')
    plt.xlabel("max_depth")
    plt.ylabel("n_estimators")
    plt.colorbar()
    
plot_scores(grid_search.grid_scores_)

from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint

parameter_grid = {'max_depth': sp_randint(1, 64),
                  'n_estimators': sp_randint(10, 160)}

random_search = RandomizedSearchCV(RandomForestClassifier(),
                                 parameter_grid,
                                 n_iter=20)

random_search.fit(X, y)

plot_scores(random_search.grid_scores_)

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


classifiers = [('Dummy', DummyClassifier(strategy='constant', constant='r')),
               ('RF', RandomForestClassifier())]
# as this is a toy example you should be able to predict the
# performance of DummyClassifier ahead of time.

for name,classifier in classifiers:
    scores = cross_val_score(classifier, X, y, scoring='accuracy')
    print(name, 'scores:', scores)

