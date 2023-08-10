import numpy as np
import scipy 
import seaborn as sns
import pandas as pd
import scipy.stats as stats

import patsy

import matplotlib
import matplotlib.pyplot as plt

get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('matplotlib inline')

plt.style.use('fivethirtyeight')

prop = pd.read_csv('../datasets/assessor_sample.csv')

prop.shape

prop.dtypes

prop.isnull().sum()

prop.property_class.unique()

prop.neighborhood.unique()

prop_samp = prop.sample(n=25000)

f = 'value ~ '+' + '.join([c for c in prop_samp.columns if not c == 'value'])+' -1'
print f
y, X = patsy.dmatrices(f, data=prop_samp, return_type='dataframe')
y = y.values.ravel()

print y.shape, X.shape

# I am predicting the value of the house from the other variables.

from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# standardize the predictors
ss = StandardScaler()
Xs = ss.fit_transform(X)
print Xs.shape

# set up my gridsearch parameters:
sgd_params = {
    'loss':['squared_loss','huber'],
    'penalty':['l1','l2'],
    'alpha':np.logspace(-5,1,25)
}

sgd_reg = SGDRegressor()
sgd_reg_gs = GridSearchCV(sgd_reg, sgd_params, cv=5, verbose=False)

np.logspace(-5,1,25)

np.linspace(-5,1,25)

# SGD is pretty fast compared to other sklearn solvers - but can still
# take a good long while depending on the gridsearch and the size of
# the dataset.
sgd_reg_gs.fit(Xs, y)

print sgd_reg_gs.best_params_
print sgd_reg_gs.best_score_
sgd_reg = sgd_reg_gs.best_estimator_

# We can see that the gradient descent got a .22 R2 using the ridge and squared loss.

value_coefs = pd.DataFrame({'coef':sgd_reg.coef_,
                            'mag':np.abs(sgd_reg.coef_),
                            'pred':X.columns})
value_coefs.sort_values('mag', ascending=False, inplace=True)
value_coefs.iloc[0:10, :]

prop_samp.columns

prop_samp.year_built.value_counts()

# lets see if we can predict if a house was built past 1980
prop_samp['built_past1980'] = prop_samp.year_built.map(lambda x: 1 if x >= 1980 else 0)

# make the target and calculate the baseline:
y = prop_samp.built_past1980.values
print 1. - np.mean(y)

f = '''
~ baths + beds + lot_depth + basement_area + front_ft + owner_pct +
rooms + property_class + neighborhood + tax_rate + volume + sqft + stories +
zone + value -1
'''

X = patsy.dmatrix(f, data=prop_samp, return_type='dataframe')

Xs = ss.fit_transform(X)
print y.shape, Xs.shape

sgd_cls_params = {
    'loss':['log'],
    'penalty':['l1','l2'],
    'alpha':np.logspace(-5,2,50)
}

sgd_cls = SGDClassifier()
sgd_cls_gs = GridSearchCV(sgd_cls, sgd_cls_params, cv=5, verbose=1)

sgd_cls_gs.fit(Xs, y)

print sgd_cls_gs.best_params_
print sgd_cls_gs.best_score_
sgd_cls = sgd_cls_gs.best_estimator_



