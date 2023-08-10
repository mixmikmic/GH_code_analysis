import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

row_data = pd.read_csv('./bs_station_a.csv')

row_data.bs.plot()

row_data.head()

row_data.describe()



sns.pairplot(row_data)

from sklearn import linear_model

sklearn.linear_model

clf = linear_model.LinearRegression()
# 'id', 'day', 'year', 'yday', 'wday', 'Tmin0', 'Tmean1', 'Tmean2', 'Tmax0', 'rainfall', 'x', 'y', 'h', 'AR_start', 'month'
X = row_data[['Tmin0', 'rainfall']].as_matrix()
Y = row_data.bs.as_matrix()

clf.fit(X, Y)

print('coef: {}'.format(clf.coef_))
print('intercept: {}'.format(clf.intercept_))
print('R: {}'.format(clf.score(X, Y)))

clf.rank_

clf.singular_

import numpy as np

plt.figure(figsize=(10, 6), dpi=200)
plt.plot(Y)
plt.plot(clf.predict(X))
plt.show()

data = pd.DataFrame(data=row_data[['bs', 'yday', '']])

import statsmodels.api as sm

x = sm.add_constant(X)
results = sm.OLS(endog=Y, exog=x).fit()
results.summary()





1

row_data[['bs', 'Tmin0', 'Tmean1', 'Tmean2', 'Tmax0']].corr()



