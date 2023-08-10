# Imports 
get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Statmodels & patsy
import patsy
import statsmodels.api as sm

# Linear Models with sklearn
from sklearn import linear_model

# Generate some correlated data
corr = 0.75
covs = [[1, corr], [corr, 1]]

means = [0, 0]

dat = np.random.multivariate_normal(means, covs, 1000)

# Check out the data we generated
plt.scatter(dat[:, 0], dat[:, 1], alpha=0.5);

# Put data into a DataFrame
df = pd.DataFrame(dat, columns=['D1', 'D2'])

# Eye ball the data
df.head()

# Patsy gives us an easy way to construct design matrices
#  For our purpose, 'design matrices' are just organized matrices of our predictor and output variables
outcome, predictors = patsy.dmatrices('D1 ~ D2', df)

# Use statsmodels to intialize the OLS model
mod = sm.OLS(outcome, predictors)

# Fit the model
res = mod.fit()

# Check out the results
print(res.summary())

# Add a new column of data to df
df['D3'] = pd.Series(np.random.randn(1000), index=df.index)
df.head()

# Predict D1 from D2 and D3
outcome, predictors = patsy.dmatrices('D1 ~ D2 + D3', df)
mod = sm.OLS(outcome, predictors)
res = mod.fit()

print(res.summary())

# Convert data into shape for easier use with sklearn
d1 = np.reshape(df.D1.values, [len(df.D1), 1])
d2 = np.reshape(df.D2.values, [len(df.D2), 1])
d3 = np.reshape(df.D3.values, [len(df.D3), 1])

# Initialize linear regression model
reg = linear_model.LinearRegression()

# Fit the linear regression model
reg.fit(d2, d1)

# Check the results of this
#  If you compare these to what we got with statsmodels above, they are indeed the same
print(reg.intercept_[0])
print(reg.coef_[0][0])

# Initialize and fit linear model
reg = linear_model.LinearRegression()
reg.fit(np.hstack([d2, d3]), d1)

# Check the results of this
#  If you compare these to what we got with statsmodels above, they are indeed the same
print('Intercept: \t', reg.intercept_[0])
print('Theta D2 :\t', reg.coef_[0][0])
print('Theta D3 :\t', reg.coef_[0][1])

