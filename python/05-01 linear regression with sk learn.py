import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model

np.random.seed(1)
x = 10 * np.random.rand(50)
y = 2 * x -1 + np.random.randn(50)
plt.scatter(x, y)
plt.show()

model = linear_model.LinearRegression(fit_intercept = True)
# creates an instance of a linear regression model where we will estimate the intercept

print(model)

print(x)

x.shape

# SciKit Learn requires that the features (x) be a matrix
# and the response y be a one-dimension array

# we create an X matrix using the x values
X = x.reshape([50,1])
print(X.shape)

# now we fit the model
model.fit(X, y)
print(model)

print(model.coef_)

print(model.intercept_)

prediction_x  = np.linspace(0, 10)
prediction_x.shape

prediction_x = prediction_x.reshape([50,1])

model.predict(prediction_x)

plt.scatter(x, y)
plt.plot(prediction_x, model.predict(prediction_x), color = 'red')
plt.show()

residuals = y - model.predict(X)

print(residuals)

np.mean(residuals)

plt.hist(residuals)

plt.plot(x, residuals, "o")

from sklearn.datasets import load_boston
boston = load_boston()

type(boston)

boston.keys()  # boston is a dictionary, the x variables are in 'data' and the y values are in 'target'

boston.data.shape  # the x matrix, 506 rows 13 columns

boston.target.shape # the one-dimensional target y values

print(boston.feature_names)  # the names of the x variables

print(boston.DESCR)  # data dictionary

type(boston.data)

# we will convert to pandas dataframe
bos = pd.DataFrame(boston.data)
print(bos.head())

bos.columns = boston.feature_names
bos.head()

boston.target[:10]

# we will add the array of values to the bos dataframe
bos['PRICE'] = boston.target

bos.head()

bosmodel = linear_model.LinearRegression()

X = bos.loc[:, 'CRIM':'LSTAT']
X.head()

type(bos.loc[:,'PRICE'])

bosmodel.fit(X, boston.target)

bosmodel.coef_

yhat = bosmodel.predict(X)

plt.plot(yhat, boston.target, 'o')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()

residuals = boston.target - yhat
plt.plot(yhat, residuals, 'o')
plt.xlabel('predicted')
plt.ylabel('residual')
plt.show()

bos.loc[bos['PRICE'] == 50, : ]

