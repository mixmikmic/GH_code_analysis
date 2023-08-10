import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

boston = load_boston()
X, y = boston.data, boston.target

regression = LinearRegression()
regression.fit(X,y)

from sklearn.feature_selection import RFECV
selector = RFECV(estimator=regression, cv=10, 
                 scoring='neg_mean_squared_error')
selector.fit(X,y)
print ('Optimal number of features: %d' %selector.n_features_)

print (boston.feature_names[selector.support_])

