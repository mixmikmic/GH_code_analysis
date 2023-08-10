

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



n = 20
x = np.linspace(0, 5, n)

sigma = 0.3
beta = np.array([1, 0.5, -0.02,5])    # real coefficient
e = np.random.normal(size=n)
X = np.column_stack((x, np.sin(x), (x-3)**2, np.ones(n))) 
y_true = np.dot(X,beta)
y = y_true + e

#do regression
model = sm.OLS(y, X)   #Pick a class. GLS, WLS...
results = model.fit()





print(results.summary())

#test beta_2 = beta_3 = 0
print(results.f_test("x2 = x3 = 0"))

#test R beta = 0
R = [[0, 1, 0, 0], [0, 0, 1, 0]]
print(results.f_test(R))





prstd, iv_l, iv_u = wls_prediction_std(results)
#wls_prediction_std returns standard deviation and 
#confidence interval of my fitted model data

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(x, y, 'o', label="data")
ax.plot(x, y_true, 'b-', label="Real")
ax.plot(x, results.fittedvalues, 'r--.', label="OLS")
ax.plot(x, iv_u, 'r--')
ax.plot(x, iv_l, 'r--')
ax.legend(loc='best')



