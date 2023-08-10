#importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# importing dataset
dataset = pd.read_csv('Salary_Data.csv')

dataset.head(n=30)

X = dataset.iloc[:, 0].values.reshape(-1, 1)
Y = dataset.iloc[:, 1].values.reshape(-1, 1)

#Viewing of dimension
X.shape, Y.shape

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Viewing of dimension
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

#Fitting Simple Linear Regression Model to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#plotting the data observation FOR TRAIN DATA with help of matplotlib
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

#plotting the data observation FOR test Data with help of matplotlib
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()



