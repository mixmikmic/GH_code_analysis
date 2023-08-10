import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

X = np.asarray(list(range(1,30)))
Y = X**2

plt.scatter(X,Y)
plt.show()

X = X.reshape(1,-1)
Y = Y.reshape(1,-1)


model = LinearRegression() # Don't get confused, we're using polynomial features this time
poly = PolynomialFeatures(degree=2).fit_transform(X)
model.fit(poly,Y)
predicted = model.predict(poly)
print(X)
print(predicted)

