from sklearn import linear_model
clf = linear_model.LinearRegression()

clf.fit([[0,0],[1,1],[2,2]],[0,1,2])

clf.coef_

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# load the datasets
diabetes = datasets.load_diabetes()

# use only one feature
diabetes_X = diabetes.data[:,np.newaxis, 2]

diabetes.data.shape # apparently 442 records, 10 features

# split data into training and testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# create a linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# coefficients
print('Coefficients: \n', regr.coef_)
# the mean square error
print('Residual sum of squares: %.2f' % np.mean((regr.predict(diabetes_X_test)-diabetes_y_test) **2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

clf = linear_model.Ridge(alpha=.5)
clf.fit([[0,0],[0,0],[1,1]], [0,0.1,1])

clf.coef_

clf.intercept_

X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])

y = np.ones(10)

# compute paths
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_)

# display results

ax = plt.gca()
ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

clf = linear_model.Lasso(alpha = 0.1)
clf.fit([[0,0],[1,1]],[0,1]) # so in our case, the target [0,1] should be a linear combination of the environment (input)

clf.predict([1,1])

clf.predict([0,0])

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X /= X.std(axis=0) # standardize data (easier to set the l1_ratio parameter)

eps = 5e-3 # the smaller it is, the longer is on the path

print("Computing regularization path using the lasso....")
alphas_lasso, coefs_lasso, _ = linear_model.lasso_path(X, y, eps, fit_intercept=False)

print("Computing regularization path using the positive lasso...")
alphas_positive_lasso, coefs_positive_lasso, _ = linear_model.lasso_path(
    X, y, eps, positive=True, fit_intercept=False)

print("Computing regularization path using the elastic net...")
alphas_enet, coefs_enet, _ = linear_model.enet_path(
    X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)

print("Computing regularization path using the positve elastic net...")
alphas_positive_enet, coefs_positive_enet, _ = linear_model.enet_path(
    X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)

coefs_lasso.T.shape

# Display results

plt.figure(1)
ax = plt.gca()
ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
l1 = plt.plot(-np.log10(alphas_lasso), coefs_lasso.T)
l2 = plt.plot(-np.log10(alphas_enet), coefs_enet.T, linestyle='--')

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and Elastic-Net Paths')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
plt.axis('tight')


plt.figure(2)
ax = plt.gca()
ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
l1 = plt.plot(-np.log10(alphas_lasso), coefs_lasso.T)
l2 = plt.plot(-np.log10(alphas_positive_lasso), coefs_positive_lasso.T,
              linestyle='--')

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and positive Lasso')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
plt.axis('tight')


plt.figure(3)
ax = plt.gca()
ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
l1 = plt.plot(-np.log10(alphas_enet), coefs_enet.T)
l2 = plt.plot(-np.log10(alphas_positive_enet), coefs_positive_enet.T,
              linestyle='--')

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Elastic-Net and positive Elastic-Net')
plt.legend((l1[-1], l2[-1]), ('Elastic-Net', 'positive Elastic-Net'),
           loc='lower left')
plt.axis('tight')
plt.show()

np.random.seed(42)

X = diabetes.data
y = diabetes.target

print("Computing regularization path using LARS")
alphas, _, coefs = linear_model.lars_path(X, y, method="lasso", verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

xx

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()

X = [[0.,0.],[1.,1.], [2.,2.],[3.,3.]]
Y = [0.,1.,2.,3.]
clf = linear_model.BayesianRidge()

clf.fit(X,Y)

clf.predict([[1,0.]]) # after being fitted, the model can be used to predict new values

# the weights w of the model can be accessed
clf.coef_

from sklearn.svm import l1_min_c

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y !=2 ]

X.shape

X -= np.mean(X,0)

cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)

from datetime import datetime
print("Computing regularisation path ...")
start = datetime.now()

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(X, y)
    coefs_.append(clf.coef_.ravel().copy())
print("This took", datetime.now() - start)

coefs_ = np.array(coefs_)

coefs_.shape

plt.plot(np.log10(cs), coefs_)
ymin, ymax = plt.ylim()
plt.xlabel('log(C)')
plt.ylabel('Coefficients')
plt.title('Logistic regression path')
plt.axis('tight')
plt.show()

n_samples = 1000
n_outliers = 50

X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1, n_informative=1, noise=10, coef=True, random_state=0)

Y

X.shape

# add outlier data
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers,1))

Y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# Fit line using all data

model = linear_model.LinearRegression()
model.fit(X, y)

# Robustly fit linear model with RANSAC algorithm

model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(X, y)

inlier_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# predict data of estimated models

line_X = np.arange(-5,5)

line_y = model.predict(line_X[:,np.newaxis])
line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])

# Compare estimated coefficients
print("Estimated coefficients (true, normal, RANSAC):")
print(coef, model.coef_, model_ransac.estimator_.coef_)

plt.plot(X[inlier_mask], y[inlier_mask], '.g', label='Inliers')
plt.plot(X[outlier_mask], y[outlieer_mask], '.r', label='Outliers')
plt.plot(line_X, line_y, '-k', label='Linear regressor')
plt.plot(line_X, line_y_ransac, '-b', label='RANSAC regressor')
plt.legend(loc='lower right')
plt.show()

from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor

estimators = [('OLS', LinearRegression()), ('Theil-Sen', TheilSenRegressor(random_state=42)), ('RANSAC', RANSACRegressor(random_state=42))]

np.random.seed(0)
n_samples = 200
# Linear model y = 3*x + N(2, 0.1**2)
x = np.random.randn(n_samples)
w = 3.
c = 2.
noise = 0.1 * np.random.randn(n_samples)
y = w * x + c + noise

# 10% outliers
y[-20:] += -20* x[-20:]
X = x[:, np.newaxis]

X.shape

x.shape

import time
plt.plot(x, y, 'k+', mew=2, ms=8)
line_x = np.array([-3, 3])
for name, estimator in estimators:
    t0 = time.time()
    estimator.fit(X, y)
    elapsed_time = time.time() - t0
    y_pred = estimator.predict(line_x.reshape(2, 1))
    plt.plot(line_x, y_pred, label='%s (fite time: %.2fs)' % (name, elapsed_time))
    
plt.axis('tight')
plt.legend(loc='upper left')
plt.show()



