from sml import execute

query = 'READ "../data/auto-mpg.csv" (separator = "\s+", header = None) AND REPLACE ("?", "mode") AND SPLIT (train = .8, test = .2, validation = .0) AND  REGRESS (predictors = [2,3,4,5,6,7,8], label = 1, algorithm = simple)'

query = 'READ "../data/auto-mpg.csv" (separator = "\s+", header = None) AND REPLACE (missing = "?", strategy = "mode") AND SPLIT (train = .8, test = .2, validation = .0) AND REGRESS (predictors = [2,3,4,5,6,7,8], label = 1, algorithm = simple) '

execute(query, verbose=True)

import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.cross_validation import train_test_split

#Names of all of the columns
names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
         'model_year','origin','car_name']

#Import dataset
data = pd.read_csv('../data/auto-mpg.csv', sep = '\s+', header = None, names = names)

data.head()

# Remove NaNs
data_clean=data.applymap(lambda x: np.nan if x == '?' else x).dropna()

# Sep Predictiors From Labels
X = data_clean[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', "origin"]]

#Select target column
y = data_clean['mpg']

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# Define and train  linear regression model
estimator = linear_model.LinearRegression()

# Train Linear Regression Model
estimator.fit(X_train, y_train)

# Prediction on Testing Set
score = estimator.score(X_test, y_test)
print('Accuracy :', score)



