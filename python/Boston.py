get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

from sklearn import datasets
boston = datasets.load_boston()

boston.keys()

boston.data.shape

boston.feature_names

print(boston.DESCR)

df = pd.DataFrame(boston.data)

df.head()

df.columns = boston.feature_names

df.head()

boston.target[:5]

from sklearn import linear_model
lm = linear_model.LinearRegression()

X = df

lm.fit(X, boston.target)

lm.intercept_

lm.coef_

pd.DataFrame(zip(X.columns, lm.coef_), columns=['features', 'coeffecients'])

plt.scatter(df.RM, boston.target)
plt.xlabel("Avg num of rooms")
plt.ylabel("Housing price")

lm.predict(X)[:5]

plt.scatter(boston.target, lm.predict(X))
plt.xlabel("Price")
plt.ylabel("Predicted Price")

