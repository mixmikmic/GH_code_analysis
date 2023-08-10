from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data
y = boston.target

print (X[0])
print (X.shape)

y[:5]

X[:5]

from sklearn.model_selection import train_test_split
import numpy as np

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Least squares regression
theta,residuals,rank,s = np.linalg.lstsq(X_train, y_train)

# Make predictions on the test data
predictions = np.dot(X_test, theta)
# Let's see the output on training data as well, to see the training error
y_true_pred = np.dot(X_train, theta)

# MSE calculation
from sklearn.metrics import mean_squared_error

print (mean_squared_error(y_test, predictions))
print (mean_squared_error(y_train, y_true_pred))

# MAE calculation
from sklearn.metrics import mean_absolute_error

print (mean_absolute_error(y_test, predictions))
print (mean_absolute_error(y_train, y_true_pred))

# R2 Score calculation
from sklearn.metrics import r2_score

print (r2_score(y_train, y_true_pred))
print (r2_score(y_test, predictions))

X = np.c_[X, np.ones(len(X))]

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Least squares regression
theta,residuals,rank,s = np.linalg.lstsq(X_train, y_train)

# Make predictions on the test data
predictions = np.dot(X_test, theta)
# Let's see the output on training data as well, to see the training error
y_true_pred = np.dot(X_train, theta)

# MSE calculation
print (mean_squared_error(y_test, predictions))
print (mean_squared_error(y_train, y_true_pred))

# MAE calculation
print (mean_absolute_error(y_test, predictions))
print (mean_absolute_error(y_train, y_true_pred))

# R2 Score calculation
print (r2_score(y_train, y_true_pred))
print (r2_score(y_test, predictions))

import scipy

### Gradient descent ###

# Objective
def f(theta, X, y, lam):
    theta = np.matrix(theta).T
    X = np.matrix(X)
    y = np.matrix(y).T
    diff = X*theta - y
    diffSq = diff.T*diff
    diffSqReg = diffSq / len(X) + lam*(theta.T*theta)
    #print ("offset =", diffSqReg.flatten().tolist())
    return diffSqReg.flatten().tolist()[0]

# Derivative
def fprime(theta, X, y, lam):
    theta = np.matrix(theta).T
    X = np.matrix(X)
    y = np.matrix(y).T
    diff = X*theta - y
    res = 2*X.T*diff / len(X) + 2*lam*theta
    #print ("gradient =", np.array(res.flatten().tolist()[0]))
    return np.array(res.flatten().tolist()[0])

theta, _, _ = scipy.optimize.fmin_l_bfgs_b(f, [0]*14, fprime, args = (X_train, y_train, 0.1))

# Make predictions on the test data
predictions = np.dot(X_test, theta)
# Let's see the output on training data as well, to see the training error
y_true_pred = np.dot(X_train, theta)

# MSE calculation
print (mean_squared_error(y_test, predictions))
print (mean_squared_error(y_train, y_true_pred))

# MAE calculation
print (mean_absolute_error(y_test, predictions))
print (mean_absolute_error(y_train, y_true_pred))

# R2 Score calculation
print (r2_score(y_train, y_true_pred))
print (r2_score(y_test, predictions))

import numpy
import random

def feature():
    return [random.random() for x in range(13)]

X_train2 = [feature() for d in X_train]
X_test2 = [feature() for d in X_test]

theta,residuals,rank,s = numpy.linalg.lstsq(X_train2, y_train)

# Make predictions on the test data
predictions = np.dot(X_test2, theta)
# Let's see the output on training data as well, to see the training error
y_true_pred = np.dot(X_train2, theta)

# MSE calculation
print (mean_squared_error(y_test, predictions))
print (mean_squared_error(y_train, y_true_pred))

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

rng = np.random.RandomState(42)
X = np.c_[X, rng.randn(X.shape[0], 14)]  # add some bad features

# normalize data as done by Lars to allow for comparison
X /= np.sqrt(np.sum(X ** 2, axis=0))

model_bic = LassoLarsIC(criterion='bic')
t1 = time.time()
model_bic.fit(X, y)
t_bic = time.time() - t1
alpha_bic_ = model_bic.alpha_

model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X, y)
alpha_aic_ = model_aic.alpha_

def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')

plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title('Information-criterion for model selection (training time %.3fs)'
          % t_bic)
plt.show()





