get_ipython().magic('matplotlib inline')

import matplotlib
from matplotlib import pyplot as plt

import seaborn

import numpy as np

from sklearn.datasets import make_classification

### Set the number of samples and dimensions
### With d = 2 we will still be able to visualize our dataset
n, d = 100, 2

np.random.seed(2)

X, y = make_classification(n_samples=n, 
                           n_features=d, 
                           n_informative=d, 
                           n_redundant=0, 
                           n_repeated=0, 
                           n_classes=2, 
                           n_clusters_per_class=1, 
                           flip_y=0.05, 
                           class_sep=1.0, 
                           )

### BEGIN STUDENT ###
# Print the shape of

print("Shape of X: {}".format(X.shape))
print("Shape of y: {}".format(y.shape))

### END STUDENT ###

### Save the colors so that we ca use them later
colors = {
    'orange':'#FF9707',
    'azure': '#00A5DD',
}

fig, ax = plt.subplots()

ax.set_title("Classification dataset")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")

X1 = X[np.argwhere(y == 1).ravel(), :]
X0 = X[np.argwhere(y == 0).ravel(), :]

ax.scatter(X0[:,0], X0[:,1], c=colors['orange'], marker='^', label='Class 1');
ax.scatter(X1[:,0], X1[:,1], c=colors['azure'], label='Class 2');

ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);

### BEGIN STUDENTS ###
# clf = ...

from sklearn.linear_model import RidgeClassifier

clf = RidgeClassifier()

clf.fit(X, y)

### END STUDENTS ###

### Save the colors so that we ca use them later
colors = {
    'orange':'#FF9707',
    'azure': '#00A5DD',
}

fig, ax = plt.subplots()

ax.set_title("Classification dataset")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")

X1 = X[np.argwhere(y == 1).ravel(), :]
X0 = X[np.argwhere(y == 0).ravel(), :]

ax.scatter(X0[:,0], X0[:,1], c=colors['orange'], marker='^', label='Class 1');
ax.scatter(X1[:,0], X1[:,1], c=colors['azure'], label='Class 2');

### Prepare ticks for separating line
xmin = X[:, 0].min()
xmax = X[:, 0].max()

### Stuff to compute separating hyperplane...
### Hint: this comes from putting <x, w> + c = 0
w1 = clf.coef_[0, 0]
w2 = clf.coef_[0, 1]
c = clf.intercept_[0]

sep = lambda x : -w1/w2*x-c/w2

ticks = np.linspace(xmin, xmax, 100)
y_ticks = sep(ticks)

ax.plot(ticks, y_ticks, ls='--', label='Separating hyperplane');

ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);

### BEGIN STUDENTS ###

### END STUDENTS ###

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X, y = make_classification(n_samples=150, 
                           n_features=d, 
                           n_informative=d, 
                           n_redundant=0, 
                           n_repeated=0, 
                           n_classes=2, 
                           n_clusters_per_class=1, 
                           flip_y=0.1, 
                           class_sep=1.0, 
                           )


### BEGIN STUDENTS ###

### END STUDENTS ###

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



