#importing libraries
import pandas as pd
import numpy as np

# importing dataset
dataset = pd.read_csv('50_Startups.csv')

dataset.head(n=5)

# Determining the Independent and Dependent Variable
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#The following lines create a class to categorical variable State
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#following line assign the last indices of X which is State
X[:, -1]=labelencoder_X.fit_transform(X[:, -1])

X

# [3] represent the index of variable you going to encode
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy varaible trap(means eliminate one dummy variable)
X = X[:, 1:]

#Splitting dataset into training and test in a 80:20 ratio
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression Model to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test set result
Y_pred = regressor.predict(X_test)

Y_pred, Y_test

#Building the optimal model using backward Elimination
#
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis =1)

X

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

