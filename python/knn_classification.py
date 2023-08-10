get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
from sklearn import neighbors

X = [[10],[8],[1],[5],[2]]
y = [1, 1, 0, 1, 0]


# Let's plot data
plt.scatter(X, y)
plt.show()


# How many neighbors should I base my decision on.
n_neighbors = 2


# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(X, y)

# Try predicting.
clf.predict([[1]])

