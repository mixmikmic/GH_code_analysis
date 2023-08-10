import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.datasets import load_boston
from sklearn import linear_model

# To set float output to 5 decimals and to suppress printing of small floating point values using scientific notations
np.set_printoptions(precision=5, suppress=True)

boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data["target"] = boston.target
n_samples = len(data)
features = data.columns[:-1]

X = data.ix[:, : -1]
y = data["target"].values

X.shape

y[0:4]

yq = np.array(y > 25, dtype=int)
print(type(yq[0:3]))

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

linear_regression = linear_model.LinearRegression(normalize=False, fit_intercept=True)
linear_regression.fit(X, y)
print("coefficietns: %s \n intercept: %0.3f" %(linear_regression.coef_, linear_regression.intercept_))



