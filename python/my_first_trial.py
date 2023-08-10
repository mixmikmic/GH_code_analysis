import pandas as pd

df = pd.read_csv('./train.csv')

df.describe()

df.columns

Y = df['SalePrice']

Y.describe()

import matplotlib.pyplot as plt
plt.clf()
Y.hist(bins=10).figure

import numpy as np
Y = np.log(Y)

Y.hist(bins=10).figure

df['SalePrice'] = np.log(df['SalePrice'])

df_subclass = pd.get_dummies(df['MSSubClass'], prefix='MSSubClass')

df.corr()

plt.clf()
df.groupby(['MSSubClass'])['SalePrice'].mean().plot.bar()
plt.show()

df = df.drop(['MSSubClass'], axis=1)

plt.clf()
df.groupby(['MSZoning'])['SalePrice'].mean().plot.bar()
plt.show()

df = df.drop(['MSZoning'], axis=1)

plt.clf()
df.plot.scatter('LotFrontage', 'SalePrice')
plt.show()

plt.clf()
df.plot.scatter('OverallQual', 'SalePrice')
plt.show()

factors = ['OverallQual']

plt.clf()
df.plot.scatter('YearBuilt', 'SalePrice')
plt.show()

factors.append('YearBuilt')

plt.clf()
df.plot.scatter('MasVnrArea', 'SalePrice')
plt.show()

plt.clf()
df.groupby(['MasVnrType'])['SalePrice'].mean().plot.bar()
plt.show()

plt.clf()
df.groupby(['MasVnrType'])['SalePrice'].std().plot.bar()
plt.show()

factors.append('MasVnrArea')

df.corr()[10:]

plt.clf()
df.plot.scatter('GrLivArea', 'SalePrice')
plt.show()

factors.append('GrLivArea')

plt.clf()
df.plot.scatter('GarageCars', 'SalePrice')
plt.show()

factors.append('GarageCars')

plt.clf()
df.plot.scatter('TotalBsmtSF', 'SalePrice')
plt.show()

factors.append('TotalBsmtSF')

from sklearn.linear_model import LinearRegression
model = LinearRegression()

df[factors].describe()

df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mean())

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[factors], Y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
print mean_squared_error(y_test, model.predict(X_test)) ** 0.5

df['SalePrice'].describe()

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=1, n_estimators=50, max_depth=4)

model.fit(X_train, y_train)

print mean_squared_error(y_test, model.predict(X_test)) ** 0.5



