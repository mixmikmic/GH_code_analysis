from pandas.tools.plotting import scatter_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from prettytable import PrettyTable
from sklearn.tree import DecisionTreeRegressor as DTR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import statsmodels.api as sm

get_ipython().magic('matplotlib inline')

df = pd.read_csv('data/balance.csv', index_col=0)

df[:2]

df['Gender']= df['Gender'].map({'Female': 0, ' Male': 1})
df['Married']= df['Married'].map({'Yes':1, 'No':0})
df['Student']= df['Student'].map({'Yes':1, 'No':0})

df.head()

scatter_matrix(df, figsize=(15,10), alpha=.4);

df = pd.get_dummies(df, prefix= '', prefix_sep= '', columns=['Ethnicity'])

df.drop('African American', axis=1, inplace=True)

df.head()

def summary_model(X, y, label='scatter'):
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    return model.summary()

def plot_model(X, y, label='Residuals - Plot'):
    model = sm.OLS(y, X).fit()
    residuals = model.outlier_test()['student_resid']
    y_hats = model.predict(X)
    
    plt.scatter(y_hats, residuals, alpha = .35, label=label)
    plt.xlabel('Predictions y_hats')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()
    
def feature_test(predict='[Independed_variable]', features=None, data=None):
    y = data[predict]
    x = sm.add_constant(data[features])
    
    model = sm.OLS(y, x)
    fit = model.fit()
    print fit.summary2()
    resid = fit.outlier_test()['student_resid']
    plt.scatter(fit.fittedvalues, resid)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    return fit


def interaction_test(y, x):
    x = sm.add_constant(x)
    
    model=sm.OLS(y, x)
    fit = model.fit()
    print fit.summary2()
    resid = fit.outlier_test()['student_resid']
    plt.scatter(fit.fittedvalues, resid)

X = df.copy()
y = X.pop('Balance')

print summary_model(X, y)
plot_model(X,y)

cols = X.columns

for col_name in cols:
    new_lcolumns = list(cols)
    new_lcolumns.remove(col_name)
    plot_model(X[new_lcolumns], y, 'Excluding from the model variable - ' + str(col_name))

y.hist();

y.hist(bins=100);

for col in cols:
    df.plot(kind='scatter', y='Balance', x=col, edgecolor='none', figsize=(10, 5))
    plt.xlabel(col)
    plt.ylabel('Balance')
    plt.show()


df.plot(kind='scatter', y='Balance', x='Limit', edgecolor='none', figsize=(12, 5))
f = DTR().fit(df[['Limit']], df['Balance'])
xval = np.arange(0, 14000, 100)
pred = [f.predict(v) for v in xval]
plt.plot(xval, pred, 'r-')
plt.grid('on')

xval[np.argwhere(np.array(pred) > 0)[:,0]]


df.plot(kind='scatter', y='Balance', x='Rating', edgecolor='none', figsize=(12, 5))
f = DTR().fit(df[['Rating']], df['Balance'])
xval = np.arange(0, 1000, 10)
pred = [f.predict(v) for v in xval]
plt.plot(xval, pred, 'r-')
plt.grid('on')

xval[np.argwhere(np.array(pred) > 0)[:,0]]

df[:2]

limit_threshold = (df.Limit >= 3300) & (df.Rating >=250)
df_threshold = df[limit_threshold]

X_new = df_threshold.copy()
y_new = X_new.pop('Balance')

X_new.columns
all_features= [u'Income', u'Limit', u'Rating', u'Cards', u'Age', u'Education',
       u'Gender', u'Student', u'Married', u'Asian', u'Caucasian']

feature_test(predict='Balance', features=all_features, data=df_threshold)

print summary_model(X_new, y_new)
plot_model(X_new, y_new)

scatter_matrix(X_new, figsize=(16,12), alpha=.36);

X_new[:2].T

matrizhecf = X_new.as_matrix(columns=all_features)

pretty = PrettyTable(['Field', 'VIF'])
for i in xrange(matrizhecf.shape[1]):
    pretty.add_row((X_new.columns[i], VIF(matrizhecf, i)))

print pretty

X_new1 = X_new.copy()
X_new1.pop('Rating')
X_new1[:2].T

X_new1.columns

features_1= [u'Income', u'Limit', u'Cards', u'Age', u'Education', u'Gender',
       u'Student', u'Married', u'Asian', u'Caucasian']

feature_test(predict='Balance', features=features_1, data=df_threshold)

print summary_model(X_new1, y_new)
plot_model(X_new1, y_new)

matriz1 = X_new1.as_matrix(columns=features_1)

pretty1 = PrettyTable(['Field', 'VIF'])
for i in xrange(matriz1.shape[1]):
    pretty1.add_row((X_new1.columns[i], VIF(matriz1, i)))

print pretty1

scatter_matrix(X_new1, figsize=(12, 9), alpha=.35);

drop_f = [u'Education', u'Gender', u'Married', u'Asian', u'Caucasian']

X_new2 = X_new1.copy()

X_new2.drop(drop_f, axis=1, inplace=True)

X_new2.columns
features_3 = [u'Income', u'Limit', u'Cards', u'Age', u'Student']

feature_test(predict='Balance', features=features_3, data=df_threshold)

print summary_model(X_new2, y_new)
plot_model(X_new2, y_new)

matriz2 = X_new1.as_matrix(columns=features_3)

pretty2 = PrettyTable(['Field', 'VIF'])
for i in xrange(matriz2.shape[1]):
    pretty2.add_row((X_new2.columns[i], VIF(matriz2, i)))

print pretty2

scatter_matrix(X_new2, figsize=(12, 9), alpha=.35);

























