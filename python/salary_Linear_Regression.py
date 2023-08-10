import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

salary_data = pd.read_csv("Position_Salaries.csv")

salary_data

features = salary_data.iloc[:, 1:-1].values
labels = salary_data.iloc[:, 2].values

print(features)
print(labels)

plt.scatter(features, labels)
plt.xlabel("Level in firm.")
plt.ylabel("Salary for that position.")
plt.show()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(features, labels)

plt.scatter(features, labels)
plt.plot(features, lr.predict(features), c='r')
plt.xlabel("Level in firm.")
plt.ylabel("Salary for that position.")
plt.show()

print("R^2 score for Linear Regresion on this dataset: ", lr.score(features, labels))

from sklearn.preprocessing import PolynomialFeatures

polynomial_reg = PolynomialFeatures(degree=3)

polynomial_features = polynomial_reg.fit_transform(features)

polynomial_features

polynomial_reg.fit(polynomial_features, labels)

poly_reg_new = LinearRegression()
poly_reg_new.fit(polynomial_features, labels)

plt.scatter(features, labels)
plt.plot(features, poly_reg_new.predict(polynomial_features), c='r')
plt.xlabel("Level in firm.")
plt.ylabel("Salary for that position.")
plt.show()

print("R^2 score for Linear Regresion on this dataset: ", poly_reg_new.score(polynomial_features, labels))

