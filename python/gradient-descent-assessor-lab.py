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

prop = pd.read_csv('./datasets/assessor_sample.csv')

# A:
prop.shape

prop.info()

prop_samp = prop.sample(n=25000)

f = 'value ~ ' + ' + '.join([c for c in prop_samp.columns if not c == 'value'])
print f

y, X = patsy.dmatrices(f, data=prop_samp, return_type='dataframe')
y = y.values.ravel()

print y.shape, X.shape

from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

sgd_params = {
    'loss':['squared_loss','huber'],
    'penalty':['l1','l2'],
    'alpha':np.logspace(-5,1,25)
}

sgd_reg = SGDRegressor()
sgd_reg_gs = GridSearchCV(sgd_reg, sgd_params, cv=5, verbose=False)

sgd_reg_gs.fit(Xs, y)

print(sgd_reg_gs.best_params_)
print(sgd_reg_gs.best_score_)
# get the best model
sgd_reg = sgd_reg_gs.best_estimator_

value_coefs = pd.DataFrame({'coef':sgd_reg.coef_,
                            'mag':np.abs(sgd_reg.coef_),
                            'pred':X.columns})
value_coefs.sort_values('mag', ascending=False, inplace=True)
value_coefs.iloc[0:10, :]

# A:
prop_samp.columns

prop_samp['year_built'].value_counts()

# lets see if we can predict if a house was built past 1980
prop_samp['built_past1980'] = prop_samp.year_built.map(lambda x: 1 if x >= 1980 else 0)

# make the target and calculate the baseline:
y = prop_samp.built_past1980.values
print 1. - np.mean(y)

f = '''
~ baths + beds + lot_depth + basement_area + front_ft + owner_pct +
rooms + property_class + neighborhood + tax_rate + volume + sqft + stories +
zone + value
'''

X = patsy.dmatrix(f, data=prop_samp, return_type='dataframe')

Xs = scaler.fit_transform(X)
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

value_coefs = pd.DataFrame({'coef':sgd_cls.coef_[0],
                            'mag':np.abs(sgd_cls.coef_[0]),
                            'pred':X.columns})
value_coefs.sort_values('mag', ascending=False, inplace=True)
value_coefs.iloc[0:10, :]



