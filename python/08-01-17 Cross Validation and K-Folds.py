import numpy as np

from sklearn.datasets import load_boston
boston = load_boston()
X, y = boston.data, boston.target

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X,y)

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

crossvalidation = KFold(n=X.shape[0], n_folds=10, random_state=1,)
scores = cross_val_score(regression, X, y, 
                          scoring ='neg_mean_squared_error', cv = crossvalidation, 
                         n_jobs =1)

print ('Folds: %i, mean squared error: %.2f std: %.2f' 
       % (len(scores),np.mean(np.abs(scores)),np.std(scores)))

