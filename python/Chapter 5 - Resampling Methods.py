import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures as poly

get_ipython().run_line_magic('matplotlib', 'inline')

auto = pd.read_csv('../../data/Auto.csv', na_values='?')
auto.dropna(inplace=True)

X = auto['horsepower'].values.reshape(-1, 1)
y = auto['mpg'].values

#Figure 5.2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

deg = np.arange(1, 11)
lr = LinearRegression()

#left paneel
mse = []
for d in deg:
    p = poly(d)
    X_poly = p.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.5, random_state=0)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse.append(mean_squared_error(y_test, y_pred))
ax1.plot(deg, mse, '-ro')
ax1.set(xlabel='Degree of Polynomial', ylabel='Mean Squared Error')

#right panel
#hard coding the random states because rs = 5 or 9 produce very bad plots (this is bad practice)
rs=[0, 1, 2, 3, 4, 6, 7, 8, 10, 12]
for i in range(10):
    mse = []
    for d in deg:
        p = poly(d)
        X_poly = p.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.5, random_state=rs[i])
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        mse.append(mean_squared_error(y_test, y_pred))
    ax2.plot(deg, mse, label=str(i))
ax2.set(xlabel='Degree of Polynomial', ylabel='Mean Squared Error')

from sklearn.model_selection import LeaveOneOut
from sklearn.cross_validation import KFold

deg = np.arange(1, 11)
lr = LinearRegression()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#left panel
mse = []
loo = LeaveOneOut()
for d in deg:
    p = poly(d)
    X_poly = p.fit_transform(X)
    loo_mse = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X_poly[train_index], X_poly[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        loo_mse.append((y_test - y_pred)**2)
    mse.append(np.mean(loo_mse))
ax1.plot(deg, mse, '-bo')
ax1.set(xlabel='Degree of Polynomial', ylabel='Mean Squared Error', title='LOOCV')
    
#right panel
for i in range(9):
    mse = []
    for d in deg:
        p = poly(d)
        X_poly = p.fit_transform(X)
        kf = KFold(len(X), n_folds=10, shuffle=True, random_state=i)
        kf_mse = []
        for train_index, test_index in kf:
            X_train, X_test = X_poly[train_index], X_poly[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            kf_mse.append(((y_test - y_pred)**2).mean())
        mse.append(np.mean(kf_mse))
    ax2.plot(deg, mse)
ax2.set(xlabel='Degree of Polynomial', ylabel='Mean Squared Error', title='10-Fold CV');



