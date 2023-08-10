import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
from sklearn import datasets, linear_model

# Data
diabetes = datasets.load_diabetes()
print("Observations: ", len(diabetes.target))

# One Variable (first-one)
diabetes_X = diabetes.data[:, np.newaxis, 0]
print("Sample x variable:")
print(diabetes_X[:5])

diabetes_y = diabetes.target.reshape(-1, 1)
print("Sample y variable")
print(diabetes_y[:5])

# Linear Regression
linreg = linear_model.LinearRegression()

# Fit
linreg.fit(diabetes_X, diabetes_y)

# The coefficients
print("Beta Hat: \n", linreg.coef_)
print("Intercept: \n", linreg.intercept_)

# Prediction
pred = linreg.predict(diabetes_X)

# Plot classification boundary
plt.scatter(diabetes_X, diabetes_y, color = 'grey')
plt.plot(diabetes_X, pred, color = 'black')
plt.show()

class CustomLinearRegression(object):
    
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        self.nrow = X.shape[0]
        if self.nrow != Y.shape[0]:
            raise Exception("Number of observations differs between X and y")
        # Add column of 1s for intercept coefficient
        self.X = np.concatenate((np.ones((self.nrow,1)), X), axis=1)
        self.beta_hat = np.dot(np.linalg.inv(np.dot(self.X.T,self.X)),
                               np.dot(self.X.T,Y))

    def predict(self, X):
        return np.dot(np.concatenate((np.ones((self.nrow,1)), X), axis=1), self.beta_hat)

# Linear Regression
custlinreg = CustomLinearRegression()

# Fit
custlinreg.fit(diabetes_X, diabetes_y)

# The coefficients
print("Coefficients: \n", custlinreg.beta_hat)

# Prediction
pred = custlinreg.predict(diabetes_X)

# Plot classification boundary
plt.scatter(diabetes_X, diabetes_y, color = 'grey')
plt.plot(diabetes_X, pred, color = 'black')
plt.show()

class CustomSGD(object):
    
    def __init__(self):
        pass
    
    def fit(self, X, Y, lr, epochs):
        self.nrow = X.shape[0]
        if self.nrow != Y.shape[0]:
            raise Exception("Number of observations differs between X and y")
        # Add column of 1s for intercept coefficient
        self.X = np.concatenate((np.ones((self.nrow,1)), X), axis=1)
        self.theta = np.ones((self.X.shape[-1], 1))
        
        for epoch in range(epochs):
            residual = np.dot(self.X, self.theta) - Y
            #cost = np.sum(residual**2) / (2*self.nrow)
            #print("Epoch %d | Cost: %f" % (epoch, cost))
            grad = np.dot(self.X.T, residual) / self.nrow
            # Backwards
            self.theta = self.theta - (lr*grad)

    def predict(self, X):
        return np.dot(np.concatenate((np.ones((self.nrow,1)), X), axis=1), self.theta)

# Linear Regression
custlinregsgd = CustomSGD()

# Fit
custlinregsgd.fit(diabetes_X, diabetes_y, lr=0.5, epochs=100000)

# The coefficients
print("Coefficients: \n", custlinregsgd.theta)

# Prediction
pred = custlinregsgd.predict(diabetes_X)

# Plot classification boundary
plt.scatter(diabetes_X, diabetes_y, color = 'grey')
plt.plot(diabetes_X, pred, color = 'black')
plt.show()

