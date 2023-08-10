from sml import execute

query = 'READ "../data/boston.csv" (separator = "\s+", header = 0) AND         SPLIT (train = .8, test = .2, validation = .0) AND         REGRESS (predictors = [1,2,3,4,5,6,7,8,9,10,11,12,13], label = 14, algorithm = elastic)'

execute(query, verbose=True)

import pandas as pd
import numpy as np

from sklearn import cross_validation, metrics
from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn import cross_validation, metrics

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', ' MEDV']
f = pd.read_csv("../data/boston.csv", sep = "\s+", header = None, names=names)
f.head()


features = f.drop('LSTAT', 1)
labels = f['LSTAT']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2, random_state=42)

enet = ElasticNet(alpha=0.1)
enet.fit(X_train, y_train)
pred = enet.predict(X_test)
print('Accuracy:', r2_score(pred, y_test))



