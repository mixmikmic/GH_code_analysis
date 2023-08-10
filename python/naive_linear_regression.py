import warnings
import numpy as np
import pandas as pd

from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
np.random.seed(42)

import preprocess

df_train = pd.read_csv('./data/train.csv')
df_train = preprocess.preprocess(df_train)

features = list(df_train.columns)
features.remove('SalePrice')
features.remove('Id')

df_test = pd.read_csv('./data/test.csv')
df_test = preprocess.preprocess(df_test, columns_needed=features)

print('Number of columns in training: ', len(df_train.columns))
print('Number of rows in training: ', len(df_train))

with pd.option_context('display.max_rows', 3, 'display.max_columns', 300):
    display(df_train)

print('Number of columns in test: ', len(df_test.columns))
print('Number of rows in test: ', len(df_test))

with pd.option_context('display.max_rows', 3, 'display.max_columns', 300):
    display(df_test)

cols_not_in_test = set(df_train.columns) - set(df_test.columns)
print('Columns not in test: ', cols_not_in_test)
cols_not_in_train = set(df_test.columns) - set(df_train.columns)
print('Columns not in train: ', cols_not_in_train)

# condition_dummies = pd.get_dummies(df_train[['Condition1', 'Condition2']])
with pd.option_context('display.max_rows', 10, 'display.max_columns', 300):
    display(pd.get_dummies(df_train))

print(df_train['SalePrice'].describe())
sns.distplot(df_train['SalePrice'], hist=True, kde=False, norm_hist=False)
plt.xlim([0, 800000])
plt.show()

#saleprice correlation matrix
k = 20 #number of variables for heatmap
corrmat = df_train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.25)
hm = sns.heatmap(
    cm, cbar=True, annot=True, square=True, fmt='.2f', 
    annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,
    cmap="YlGnBu")
plt.show()

# Perform very Naive linear regression
from sklearn.linear_model import LinearRegression

y_train = df_train['SalePrice'].values.reshape(-1, 1)
print('y_train: ', y_train.shape)
x_train = df_train[features].values
print('x_train: ', x_train.shape)

# Train the model using the training sets
model = LinearRegression()
model.fit(x_train, y_train)

# Predictions from train set
from sklearn.metrics import mean_squared_error, r2_score

y_train_pred = model.predict(x_train)
print('y_train_pred: ', y_train_pred.shape)

print('Mean squared error: {}'.format(mean_squared_error(y_train, y_train_pred)))
print('Variance score: {}'.format(r2_score(y_train, y_train_pred)))


# Plot predictions
plt.figure(figsize=(10, 4))
plt.scatter(y_train, y_train_pred)
plt.title('Linear regression')
plt.xlabel('Real values')
plt.ylabel('Predicted values')
plt.show()


# Plot the residuals (errors) after fitting a linear model
plt.figure(figsize=(10, 4))
sns.residplot(np.squeeze(y_train), np.squeeze(y_train_pred), lowess=True)
plt.title('Residual plot')
plt.xlabel('Real values')
plt.ylabel('Residuals')
plt.show()

# Make predictions on test set and save the results
# This gets only an error of 0.24604 on the leaderboard. (Already above benchmark of 0.40890)
x_test = df_test[features].values
print('x_test: ', x_test.shape)

y_test_pred = model.predict(x_test)
y_test_pred = np.clip(y_test_pred, a_min=0, a_max=None)
print('y_test_pred: ', y_test_pred.shape)

# Test for negative values:
assert (y_test_pred >= 0).all()
print('{} values above 800k'.format(int(sum(y_test_pred >= 800000))))

df_test_predict = df_test[['Id']]
df_test_predict['SalePrice'] = np.squeeze(y_test_pred)
assert df_test_predict.notnull().all().all()

with pd.option_context('display.max_rows', 3, 'display.max_columns', 2):
    display(df_test_predict)
    
df_test_predict.to_csv('output_naive.csv', index=False)

sns.distplot(np.squeeze(y_test_pred), hist=True, kde=False, norm_hist=False)
plt.title('Test predicted SalePrice histogram')
plt.ylabel('Count')
plt.xlim([0, 700000])
plt.show()

