get_ipython().magic('matplotlib inline')

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression


np.random.seed(6450345)

def make_data(N=1000, n_vars=10,
              n_classes=2):
    X = np.random.normal(size=(N,n_vars))
    y = np.random.choice(n_classes, N)
    
    return X, y

X,y = make_data(n_vars=5)

sns.corrplot(np.c_[X, y],
             names=["var%i"%n for n in range(X.shape[1])]+['y'])

X,y = make_data(N=2000, n_vars=50000)

select3 = SelectKBest(f_regression, k=20)
X_three = select3.fit_transform(X,y)

clf = GradientBoostingClassifier()
scores = cross_val_score(clf, X_three, y, cv=5, n_jobs=8)

print "Scores on each subset:"
print scores
avg = (100*np.mean(scores), 100*np.std(scores)/np.sqrt(scores.shape[0]))
print "Average score and uncertainty: (%.2f +- %.3f)%%"%avg

from sklearn.pipeline import make_pipeline

clf = make_pipeline(SelectKBest(f_regression, k=20),
                    GradientBoostingClassifier())

scores = cross_val_score(clf, X, y, cv=5)

print "Scores on each subset:"
print scores
avg = (100*np.mean(scores), 100*np.std(scores)/np.sqrt(scores.shape[0]))
print "Average score and uncertainty: (%.2f +- %.3f)%%"%avg

