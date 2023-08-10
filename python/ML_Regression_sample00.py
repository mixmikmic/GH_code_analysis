get_ipython().magic('matplotlib inline')

import os
import requests
import pandas as pd 
import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix

from sklearn import cross_validation as cv
from sklearn.cross_validation import train_test_split as tts

from sklearn.linear_model import Ridge
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

installation = pd.read_csv('/Users/graceluningfu/Documents/505workshop/machine-learning/notebook/data/instances_feature00.csv', sep=",")

installation.head()

installation.columns = ['sfh_qty','th_qty','households','hhincome_below_10k','hhincome_10_below_15k',                  'hhincome_15_below_25','hhincome_25_below_35k','hhincome_35_below_50k','hhincome_50_below_75k',                  'hhincome_75_below_100', 'hhincome_100_below_150k','hhincome_150_below_200k','hhincome_above_200k',                 'avg_rtl_price','dem_share', 'year_since_1990', 'count']

installation.head()

installation.describe()

scatter_matrix(installation, alpha=0.2, figsize=(18,18), diagonal='kde')
plt.show()

installation_features = installation.ix[:,0:16]
installation_labels = installation.ix[:,16:]

model = RandomizedLasso(alpha=0.1)
model.fit(installation_features, installation_labels["count"])
names = list(installation_features)

print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), model.scores_), 
                 names), reverse=True))

"""for another label/outcome
model = RandomizedLasso(alpha=0.1)
model.fit(installation_features, installation_labels["total"])
names = list(installation_features)

print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), model.scores_), 
                 names), reverse=True))
"""

installation_labels = installation.ix[:,16]

splits = cv.train_test_split(installation_features, installation_labels, test_size=0.2)
X_train, X_test, y_train, y_test = splits

model = Ridge(alpha=0.1)
model.fit(X_train, y_train)

expected = y_test
predicted = model.predict(X_test)

print("Ridge Regression model")
print("Mean Squared Error: %0.3f" % mse(expected, predicted))
print("Coefficient of Determination: %0.3f" % r2_score(expected, predicted))

model = LinearRegression()
model.fit(X_train, y_train)

expected = y_test
predicted = model.predict(X_test)

print("Linear Regression model")
print("Mean Squared Error: %0.3f" % mse(expected, predicted))
print("Coefficient of Determination: %0.3f" % r2_score(expected, predicted))

model = RandomForestRegressor()
model.fit(X_train, y_train)

expected = y_test
predicted = model.predict(X_test)

print("Random Forest model")
print("Mean squared error = %0.3f" % mse(expected, predicted))
print("R2 score = %0.3f" % r2_score(expected, predicted))

