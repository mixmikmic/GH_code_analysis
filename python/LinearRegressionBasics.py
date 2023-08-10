import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('../../datasets/mpg.csv')

df.head()

df.head().transpose()

# drop the id column (unnamed)

df = df.drop('Unnamed: 0', 1)
df.head().transpose()

# build training and testing sets

train = df.sample(frac = 7/10, random_state = 1)

test = df.drop(train.index)

train.count()

test.count()

train_X = np.asarray(train[['displ', 'class']])
train_y = np.asarray(train['hwy'])

test_X = np.asarray(test[['displ', 'class']])
test_y = np.asarray(test['hwy'])

# create the linear regression model

rgr = linear_model.LinearRegression()

class_dummies = pd.get_dummies(df['class'])
class_dummies.head()

df2 = pd.concat([df, class_dummies], axis = 1)
df2.head().transpose()

# split up the dataset again to train and test.

train = df2.sample(frac = 7/10, random_state = 1)
test = df2.drop(train.index)

train_X = np.asarray(train[['displ', '2seater', 'compact', 'midsize',                             'minivan', 'pickup', 'subcompact', 'suv']])

train_y = np.asarray(train['hwy'])

test_X = np.asarray(test[['displ', '2seater', 'compact', 'midsize',                          'minivan', 'pickup', 'subcompact', 'suv']])

test_y = np.asarray(test['hwy'])

rgr.fit(train_X, train_y)

rgr.intercept_

print('Coffefients: \n', rgr.coef_)

# zip it up to see the name of the variable next to the coefficient

z = zip(['displ', '2seater', 'compact', 'midsize',          'minivan', 'pickup', 'subcompact', 'suv'], rgr.coef_)

list(z)

# looking at the R-Squared value on the training set
# Variance score = 1 is perfect prediction

print('R Squared: {}'.format(rgr.score(train_X, train_y)))

# predict the highway mpg given that we 
# know the displacement and class already

rgr.predict(test_X)

