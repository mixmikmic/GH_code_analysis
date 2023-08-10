get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('run ../src/LinearRegression.py')
get_ipython().magic('run ../src/PolynomialFeatures.py')

# LINEAR REGRESSION

# Generate random data
X = np.linspace(0,20,10)[:,np.newaxis]
y = 0.1*(X**2) + np.random.normal(0,2,10)[:,np.newaxis] + 20

# Fit model to data
lr = LinearRegression()
lr.fit(X,y)

# Predict new data
x_test = np.array([0,20])[:,np.newaxis]
y_predict = lr.predict(x_test)


# POLYNOMIAL REGRESSION

# Fit model to data
poly = PolynomialFeatures(2)
lr = LinearRegression()
lr.fit(poly.fit_transform(X),y)

# Predict new data
x_pol = np.linspace(0, 20, 100)[:, np.newaxis]
y_pol = lr.predict(poly.fit_transform(x_pol))

# Plot data

fig = plt.figure(figsize=(14, 6))

# Plot linear regression
ax1 = fig.add_subplot(1, 2, 1)
plt.scatter(X,y)
plt.plot(x_test, y_predict, "r")
plt.xlim(0, 20)
plt.ylim(0, 50)

# Plot polynomial regression
ax2 = fig.add_subplot(1, 2, 2)
plt.scatter(X,y)
plt.plot(x_pol, y_pol, "r")
plt.xlim(0, 20)
plt.ylim(0, 50);

get_ipython().magic('run ../src/LogisticRegression.py')

X = np.hstack((np.random.normal(90, 2, 100), np.random.normal(110, 2, 100)))[:, np.newaxis]
y = np.array([0]*100 + [1]*100)[:, np.newaxis]

logr = LogisticRegression(learnrate=0.002, eps = 0.001)

logr.fit(X, y)

x_test = np.array([-logr.w[0]/logr.w[1]]).reshape(1,1) #np.linspace(-10, 10, 30)[:, np.newaxis]
y_probs = logr.predict_proba(x_test)[:, 0:1]
print("Probability:" + str(y_probs))

# Plot data

fig = plt.figure(figsize=(14, 6))

# Plot sigmoid function
ax1 = fig.add_subplot(1, 2, 1)
t = np.linspace(-15,15,100)
plt.plot(t, logr._sigmoid(t))

# Plot logistic regression
ax2 = fig.add_subplot(1, 2, 2)
plt.scatter(X, y)
plt.scatter(x_test, y_probs, c='r')

get_ipython().magic('run ../src/KNearestNeighbors.py')

# Generate data from 3 gaussians
gaussian_1 = np.random.multivariate_normal(np.array([1, 0.0]), np.eye(2)*0.01, size=100)
gaussian_2 = np.random.multivariate_normal(np.array([0.0, 1.0]), np.eye(2)*0.01, size=100)
gaussian_3 = np.random.multivariate_normal(np.array([0.1, 0.1]), np.eye(2)*0.001, size=100)
X = np.vstack((gaussian_1, gaussian_2, gaussian_3))
y = np.array([1]*100 + [2]*100 + [3]*100)

# Fit the model
knn = KNearestNeighbors(5)
knn.fit(X, y)

# Predict various points in space
XX, YY = np.mgrid[-5:5:.2, -5:5:.2]
X_test = np.hstack((XX.ravel()[:, np.newaxis], YY.ravel()[:, np.newaxis]))
y_test = knn.predict(X_test)

fig = plt.figure(figsize=(14, 6))

# Plot original data
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(X[y == 1,0], X[y == 1,1], 'bo')
ax1.plot(X[y == 2,0], X[y == 2,1], 'go')
ax1.plot(X[y == 3,0], X[y == 3,1], 'ro')

# Plot predicted data
ax2 = fig.add_subplot(1, 2, 2)
ax2.contourf(XX, YY, y_test.reshape(50,50));



