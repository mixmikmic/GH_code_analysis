get_ipython().magic('matplotlib inline')
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy import stats
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from statsmodels.base.model import GenericLikelihoodModel



data = sm.datasets.spector.load_pandas()  #use the dataset spector
exog = data.exog
endog = data.endog
print(sm.datasets.spector.NOTE)
print(data.exog.head())

exog1 = sm.add_constant(exog, prepend=True)   #combine X matrix with constant

#plug in the log-likelihood function of my own model
class MyProbit(GenericLikelihoodModel):
    def loglike(self, params):
        exog = self.exog
        endog = self.endog
        q = 2 * endog - 1
        y = stats.norm.logcdf(q*np.dot(exog, params)).sum()   
        return y

sm_probit_manual = MyProbit(endog, exog1).fit()
print(sm_probit_manual.summary())



