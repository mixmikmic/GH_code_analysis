# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi

from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8

# HIDDEN
def df_interact(df, nrows=7, ncols=7):
    '''
    Outputs sliders that show rows and columns of df
    '''
    def peek(row=0, col=0):
        return df.iloc[row:row + nrows, col:col + ncols]
    if len(df.columns) <= ncols:
        interact(peek, row=(0, len(df) - nrows, nrows), col=fixed(0))
    else:
        interact(peek,
                 row=(0, len(df) - nrows, nrows),
                 col=(0, len(df.columns) - ncols))
    print('({} rows, {} columns) total'.format(df.shape[0], df.shape[1]))

# HIDDEN
ice = pd.read_csv('icecream.csv')

# HIDDEN
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

trans_ten = PolynomialFeatures(degree=10)
X_ten = trans_ten.fit_transform(ice[['sweetness']])
y = ice['overall']

clf_ten = LinearRegression(fit_intercept=False).fit(X_ten, y)

# HIDDEN
np.random.seed(1)
x_devs = np.random.normal(scale=0.4, size=len(ice))
y_devs = np.random.normal(scale=0.4, size=len(ice))

plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.scatter(ice['sweetness'], ice['overall'])
xs = np.linspace(3.5, 12.5, 1000).reshape(-1, 1)
ys = clf_ten.predict(trans_ten.transform(xs))
plt.plot(xs, ys)
plt.title('Degree 10 polynomial fit')
plt.ylim(3, 7);

plt.subplot(122)
ys = clf_ten.predict(trans_ten.transform(xs))
plt.plot(xs, ys)
plt.scatter(ice['sweetness'] + x_devs,
            ice['overall'] + y_devs,
            c='g')
plt.title('Degree 10 poly, second set of data')
plt.ylim(3, 7);

# HIDDEN
ice = pd.read_csv('icecream.csv')
transformer = PolynomialFeatures(degree=2)
X = transformer.fit_transform(ice[['sweetness']])
  
clf = LinearRegression(fit_intercept=False).fit(X, ice[['overall']])
xs = np.linspace(3.5, 12.5, 300).reshape(-1, 1)
rating_pred = clf.predict(transformer.transform(xs))

#ice.plot.scatter('sweetness', 'overall')
#plt.plot(xs, rating_pred)
#plt.title('Degree 2 polynomial fit');

temp = pd.DataFrame(xs, columns = ['sweetness'])
temp['overall'] = rating_pred

np.random.seed(42)
x_devs = np.random.normal(scale=0.2, size=len(temp))
y_devs = np.random.normal(scale=0.2, size=len(temp))
temp['sweetness'] = np.round(temp['sweetness'] + x_devs, decimals=2)
temp['overall'] = np.round(temp['overall'] + y_devs, decimals=2)

ice = pd.concat([temp, ice])

ice

# HIDDEN
plt.scatter(ice['sweetness'], ice['overall'], s=10)
plt.title('Ice Cream Rating vs. Sweetness')
plt.xlabel('Sweetness')
plt.ylabel('Rating');

from sklearn.model_selection import train_test_split

valid_size, test_size = 46, 46

# To perform a three-way split, we need to call train_test_split twice - once
# for the test set, one for the validation set.
X, X_test, y, y_test = train_test_split(
    ice[['sweetness']], ice['overall'], test_size=test_size)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=valid_size)

print(f'  Training set size: {len(X_train)}')
print(f'Validation set size: {len(X_valid)}')
print(f'      Test set size: {len(X_test)}')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# First, we add polynomial features to X_train
transformers = [PolynomialFeatures(degree=deg)
                for deg in range(1, 11)]
X_train_polys = [transformer.fit_transform(X_train)
                 for transformer in transformers]

# Display the X_train with degree 5 polynomial features
X_train_polys[4]

# Next, we train a linear regression classifier for each featurized dataset.
# We set fit_intercept=False since the PolynomialFeatures transformer adds the
# bias column for us.
clfs = [LinearRegression(fit_intercept=False).fit(X_train_poly, y_train)
        for X_train_poly in X_train_polys]

def mse_cost(y_pred, y_actual):
    return np.mean((y_pred - y_actual) ** 2)

# To make predictions on the validation set we need to add polynomial features
# to the validation set too.
X_valid_polys = [transformer.fit_transform(X_valid)
                 for transformer in transformers]

predictions = [clf.predict(X_valid_poly)
               for clf, X_valid_poly in zip(clfs, X_valid_polys)]

costs = [mse_cost(pred, y_valid) for pred in predictions]

# HIDDEN
cv_df = pd.DataFrame({'Validation Error': costs}, index=range(1, 11))
cv_df.index.name = 'Degree'
pd.options.display.max_rows = 20
display(cv_df)
pd.options.display.max_rows = 7

# HIDDEN
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(cv_df.index, cv_df['Validation Error'])
plt.scatter(cv_df.index, cv_df['Validation Error'])
plt.title('Validation Error vs. Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('Validation Error');

plt.subplot(122)
plt.plot(cv_df.index, cv_df['Validation Error'])
plt.scatter(cv_df.index, cv_df['Validation Error'])
plt.ylim(0.0427, 0.05)
plt.title('Zoomed In')
plt.xlabel('Polynomial Degree')
plt.ylabel('Validation Error')

plt.tight_layout();

best_trans = transformers[1]
best_clf = clfs[1]

training_error = mse_cost(best_clf.predict(X_train_polys[1]), y_train)
validation_error = mse_cost(best_clf.predict(X_valid_polys[1]), y_valid)
test_error = mse_cost(best_clf.predict(best_trans.transform(X_test)), y_test)

print('Degree 2 polynomial')
print(f'  Training error: {training_error:0.3f}')
print(f'Validation error: {validation_error:0.3f}')
print(f'      Test error: {test_error:0.3f}')

