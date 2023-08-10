from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from pprint import pprint

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('matplotlib inline')

plt.style.use('fivethirtyeight')

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()

# A:
print(boston.DESCR)

# A:
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

df.head()

df.info()

df.isnull().sum()

df.describe()

# Correlation
plt.figure(figsize=(15,10))
sns.heatmap(df.corr()**2, annot=True)
plt.show()

# select 4 predictors
sns.heatmap(df[['RM','PTRATIO','AGE','CRIM']].corr()**2, annot=True)

lm = LinearRegression()
X = df[['RM','PTRATIO','AGE','CRIM']]
y = df[['MEDV']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 142, test_size = 0.5)

model = lm.fit(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

# plot predictions
predictions = model.predict(X_test)
plt.figure(figsize=(15,10))
sns.jointplot(y_test.values, predictions)

# A:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 142, test_size = 0.1)

model = lm.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

predictions = model.predict(X_test)
plt.figure(figsize=(15,10))
sns.jointplot(y_test.values, predictions)

# A:
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn import metrics

for folds in range(5,11):
    print '------------------------------------\n'
    print 'K:', folds
    model = LinearRegression()
    
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=folds)
    print "Cross-validated scores:", scores
    print "Mean CV R2:", np.mean(scores)
    print 'Std CV R2:', np.std(scores)
    
    # Make cross-validated predictions
    predictions = cross_val_predict(model, X, y, cv=folds)
    print(len(predictions))
    r2 = metrics.r2_score(y, predictions)
    print "Cross-Predicted R2:", r2

# A:
from itertools import combinations

combs = []
for i in range(1, len(boston.feature_names)+1):
    for c in combinations(boston.feature_names.tolist(), i):
        combs.append(c)

print(len(boston.feature_names))
print(len(combs))

r2_score={}
linregmodel = LinearRegression()
for i,c in enumerate(combs):
    subX = df[list(c)]
    r2_score[c] = np.mean(cross_val_score(linregmodel, subX, y))

r2_score = r2_score.items()
r2_score = sorted(r2_score, key=lambda x:x[1], reverse=True)

pprint(r2_score[:5])

# A:
# These scores can be just by chance since the data is a random sample from overall population
# There is a chance that some variables might not make sense to be included as a feature.

import patsy

# A:

sns.pairplot(df[boston.feature_names.tolist()])

# predict LSTAT with NOX, RM, AGE, DIS
formula = 'LSTAT ~ NOX + RM + AGE + DIS + MEDV'
y, X = patsy.dmatrices(formula, data=df, return_type='dataframe')

# split to training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

lm = LinearRegression()
model = lm.fit(X_train, y_train)

print'Training R2 Score: ', model.score(X_train, y_train)
print'Test R2 Score: ', model.score(X_test, y_test)

# plot the predictions against test
predictions = model.predict(X_test)
r2_score = metrics.r2_score(y_test, predictions)
# print(len(y_test))
# print(len(predictions))
print(r2_score)

sns.jointplot(x=y_test.values, y=predictions)

import statsmodels.api as sm

model = sm.OLS(y_train, X_train).fit()
model.summary()

# from the summary, RM should not be used
X_train = X_train.drop(labels='NOX', axis=1)

model = sm.OLS(y_train, X_train).fit()
model.summary()

X_test = X_test.drop(labels='NOX',axis=1)

yhat = model.predict(X_test)
y_true = y_test.values.ravel()
sns.jointplot(x=y_true, y=yhat.values)
print(metrics.r2_score(y_true, yhat.values))

# Cross validation
linregmodel = LinearRegression()
kf_shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(linregmodel, X[X_test.columns.tolist()], y, cv=kf_shuffle)
predictions = cross_val_predict(linregmodel, X[X_test.columns.tolist()], y, cv=kf_shuffle)

# Very strong Model hahahah
print(scores)
print(scores.mean())
print(scores.std())

# plot 
sns.jointplot(x=y.values, y=predictions)



