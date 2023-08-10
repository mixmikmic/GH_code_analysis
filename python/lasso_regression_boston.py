import pandas as pd
import numpy as py
from sklearn.datasets import load_boston
from sklearn.linear_model import LassoLarsCV
from sklearn.cross_validation import train_test_split
import sklearn.metrics

boston = load_boston()
# the boston dataset consists of two arrays: features (.data) & target (.target)
# lets add the feature into a dataframe
df=pd.DataFrame(boston.data, columns=boston.feature_names)
df.head(2)

print boston['DESCR']

# Add housing price (target) in
df['PRICE']=pd.DataFrame(boston.target)
df.head(2)

# standardise the means to 0 and standard error to 1
from sklearn import preprocessing
for i in df.columns[:-1]: # df.columns[:-1] = dataframe for all features
    df[i] = preprocessing.scale(df[i].astype('float64'))

df.describe()

feature = df[df.columns[:-1]]
target = df['PRICE']

train_feature, test_feature, train_target, test_target = train_test_split(feature, target, random_state=123, test_size=0.2)

print train_feature.shape
print test_feature.shape

# Fit the LASSO LAR regression model
# cv=10; use k-fold cross validation
# precompute; True=model will be faster if dataset is large
model=LassoLarsCV(cv=10, precompute=False).fit(train_feature,train_target)
model

# print regression coefficients and sort them
df2=pd.DataFrame(model.coef_, index=feature.columns)
df2.sort_values(by=0,ascending=False)

# alternatively, can do this:
# dict creates dictionary; zip creates list
# coeff=dict(zip(feature.columns, model.coef_))

# LSTAT is the most important predictor, followed by RM, DIS, and RAD

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

# visualise how much each regression coefficient change to 0 when alpha is increased
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
# Transposing coefficient path
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# first coefficient that is placed is the most important

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()

# path of mean square error when alpha increases
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')

# removing more predictors does not always lead to reduction of mean square error

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(train_target, model.predict(train_feature))
test_error = mean_squared_error(test_target, model.predict(test_feature))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# MSE closer to 0 are better
# test dataset is less accurate as expected

# R-square from training and test data
rsquared_train=model.score(train_feature,train_target)
rsquared_test=model.score(test_feature,test_target)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)

# test data explained 65% of the predictors

