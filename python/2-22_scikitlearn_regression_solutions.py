import numpy as np
from datascience import *
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import KFold

bike=Table().read_table(('data/Bike-Sharing-Dataset/day.csv'))

# reformat the date column to integers representing the day of the year, 001-366
bike['dteday'] = pd.to_datetime(bike['dteday']).strftime('%j')

# get rid of the index column
bike = bike.drop(0)

bike.show(4)

# explore the data set here

# the features used to predict riders
X = bike.drop('casual', 'registered', 'cnt')
X = X.to_df()

# the number of riders
y = bike['cnt']

# set the random seed

np.random.seed(10)

# split the data
# train_test_split returns 4 values: X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.80, test_size=0.20)

# split the data
# Returns 4 values: X_train, X_validate, y_train, y_validate

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train,
                                                    train_size=0.75, test_size=0.25)

# create a model
lin_reg = LinearRegression(normalize=True)

# fit the model
lin_model = lin_reg.fit(X_train, y_train)

print(lin_model.coef_)
print(lin_model.intercept_)

# predict the number of riders
lin_pred = lin_model.predict(X_train)

# plot the residuals on a scatter plot
plt.scatter(y_train, lin_pred)
plt.title('Linear Model (OLS)')
plt.xlabel('actual value')
plt.ylabel('predicted value')
plt.show()

def rmse(pred, actual):
    return np.sqrt(np.mean((pred - actual) ** 2))

rmse(lin_pred, y_train)

# make and fit a Ridge regression model
ridge_reg = Ridge() 
ridge_model = ridge_reg.fit(X_train, y_train)

# use the model to make predictions
ridge_pred = ridge_model.predict(X_train)

# plot the predictions
plt.scatter(y_train, ridge_pred)
plt.title('Ridge Model')
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.show()

# calculate the rmse for the Ridge model
rmse(ridge_pred, y_train)

# create and fit the model
lasso_reg = Lasso(max_iter=10000)  

lasso_model = lasso_reg.fit(X_train, y_train)

# use the model to make predictions
lasso_pred = lasso_model.predict(X_train)

# plot the predictions
plt.scatter(y_train, lasso_pred)
plt.title('LASSO Model')
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.show()

# calculate the rmse for the LASSO model
rmse(lasso_pred, y_train)

# make predictions for each model
lin_vpred = lin_model.predict(X_validate)
ridge_vpred = ridge_model.predict(X_validate)
lasso_vpred = lasso_model.predict(X_validate)

# calculate RMSE for each set of validation predictions
print("linear model rmse: ", rmse(lin_vpred, y_validate))
print("Ridge rmse: ", rmse(ridge_vpred, y_validate))
print("LASSO rmse: ", rmse(lasso_vpred, y_validate))

# make predictions for the test set using one model of your choice
final_pred = lin_model.predict(X_test)
# calculate the rmse for the final predictions
print('Test set rmse: ', rmse(final_pred, y_test))

