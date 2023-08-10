get_ipython().magic("config InlineBackend.figure_format='retina'")
get_ipython().magic('matplotlib inline')

import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["font.size"] = 14

# Generate more complicated data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, centers=20, random_state=42)
labels = ["b", "r"]
y = np.take(labels, (y < 10))

print(X[:5,:])
print()
print(y[:5])

plt.figure()
for label in labels:
    mask = (y == label)
    plt.scatter(X[mask, 0], X[mask, 1], c=label, s=40)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()

# Import the decision tree class
from sklearn.tree import DecisionTreeClassifier 

# Set hyper-parameters, to control the algorithms behaviour
# Let's experiment a bit
clf = DecisionTreeClassifier()#min_samples_leaf=10)

# Learn a model from the training data
clf.fit(X, y)

from utils import plot_surface
from ipywidgets import interact


def tree(max_depth=1):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
    clf.fit(X, y)
    plot_surface(clf, X, y)
    
interact(tree, max_depth=(1, 30))

from sklearn.ensemble import RandomForestClassifier

def forest(n_estimators=10):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=1).fit(X, y)
    plot_surface(clf, X, y)

interact(forest, n_estimators=(1,50,5))

# Questions? -> Answers!
#RandomForestClassifier?

from utils import draw_tree

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)
draw_tree(clf, ['X1', 'X2'], filled=True)

