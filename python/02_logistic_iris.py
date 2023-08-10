import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
get_ipython().magic('matplotlib inline')

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

'''please change the value of C to observe the results'''
logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each
from myfun import plot_decision_regions

plt.figure()
plot_decision_regions(X,y,logreg)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.show()

print(iris.DESCR)



