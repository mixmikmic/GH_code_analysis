get_ipython().magic('matplotlib inline')

import matplotlib
from matplotlib import pyplot as plt

import seaborn

import numpy as np

from numpy.linalg import inv

n = 10 # the number of samples

X = np.random.uniform(-1, 1, size=(n,1))

w = 1.3
c = 0.7

sigma = 0.3


y = w * X + c + np.random.normal(0, sigma**2, size=(n,1))

fig, ax = plt.subplots()

ax.set_title("Dummy dataset")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.scatter(X, y, c=["orange"]);

### BEGIN STUDENTS

# w_hat = ...

w_hat = inv(X.T.dot(X)).dot(X.T).dot(y)
### END STUDENTS

fig, ax = plt.subplots()

ax.set_title("Dummy dataset")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.scatter(X, y, c=["orange"]);

xmin = X.min()
xmax = X.max()

ymin = float(w_hat * xmin)
ymax = float(w_hat * xmax)

ax.plot([xmin, xmax], [ymin, ymax], 'k-');

### BEGIN STUDENTS ###

# X_ones= ...
# w_hat = ...

X_ones = np.hstack((np.ones(shape=(n,1)), X))

w_hat = inv(X_ones.T.dot(X_ones)).dot(X_ones.T).dot(y)
### END STUDENTS ###

fig, ax = plt.subplots()

ax.set_title("Dummy dataset")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.scatter(X, y, c=["orange"]);

xmin = X.min()
xmax = X.max()

ymin = w_hat[1] * xmin + w_hat[0]
ymax = w_hat[1] * xmax + w_hat[0]

ax.plot([xmin, xmax], [ymin, ymax], 'k-');



