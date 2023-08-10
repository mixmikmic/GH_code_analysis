import numpy as np
import pandas as pd

wage = pd.read_csv('wage1.csv')
wage.head()

import statsmodels.api as sm
y = wage.wage
X = sm.add_constant(wage.educ)

lm = sm.OLS(y, X)
lm_fit = lm.fit()

resid = y - lm_fit.predict()
resid.sum()
sst = np.dot((y-y.mean()).T, (y-y.mean()))
sse = np.dot((lm_fit.predict()-y.mean()).T, (lm_fit.predict()-y.mean()))
ssr = np.dot(resid.T, resid)
sse/sst

y1 = wage.lwage
X1 = sm.add_constant(wage.educ)

lm1 = sm.OLS(y1, X1)
lm1_fit = lm1.fit()
lm1_fit.summary()

n = float(wage.wage.count())
sigmasq = (1/(n-2)) * ssr

sigmasq

X.head()



