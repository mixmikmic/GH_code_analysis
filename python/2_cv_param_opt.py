get_ipython().magic('matplotlib inline')

from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') # for classification
    # loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error') # for regression
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accurancy')
plt.show()

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') # for classification
    loss = cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error') # for regression
    k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accurancy')
plt.show()

