import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os, sys
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.naive_bayes import GaussianNB

np.random.seed(18937)

datasource = "datasets/winequality-red.csv"

print(os.path.exists(datasource))

df = pd.read_csv(datasource).sample(frac = 1).reset_index(drop = True)
df.head()

del df["Unnamed: 0"]

df.head()

df.describe()

m = GaussianNB()

X = np.array(df.iloc[:, :-1])[:, [1, 2, 6, 9, 10]]

y = np.array(df["quality"])

sklearn.model_selection.cross_val_score(m, X, y, cv = 5)

help(np.array_split)

X_folds = np.array_split(X, 5) # split the X array from earlier into 5 equal chunks 

[i.shape for i in X_folds]

y_folds = np.array_split(y, 5)
[i.shape for i in y_folds]

m = GaussianNB()

for i in range(5):
    X_train = np.concatenate([X_folds[j] for j in range(5) if j != i])
    X_test = X_folds[i]
    y_train = np.concatenate([y_folds[j] for j in range(5) if j != i])
    y_test = y_folds[i]
    
    print("CV", i)
    print("X_train", X_train.shape)
    print("X_test", X_test.shape)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)
    
    m.fit(X_train, y_train)
    print("Score", round(m.score(X_test, y_test), 3))
    print("=====================================")
    print("\n")

def cross_val_score(model, X, y, cv = 10):
    X_folds = np.array_split(X, cv)
    Y_folds = np.array_split(y, cv)
    
    for i in range(cv):
        X_train = np.concatenate([X_folds[j] for j in range(cv) if j != i])
        X_test = X_folds[i]
        y_train = np.concatenate([Y_folds[j] for j in range(cv) if j != i])
        y_test = y_folds[i]
        model.fit(X_train, y_train)
        yield model.score(X_test, y_test)

m = GaussianNB()

print("Our CV", list(cross_val_score(m, X, y, cv = 5)))
print("sklearn CV", sklearn.model_selection.cross_val_score(m, X, y, cv = 5))

s5 = sklearn.model_selection.cross_val_score(m, X, y, cv = 5)
s10 = sklearn.model_selection.cross_val_score(m, X, y, cv = 10)

print("5 fold mean", np.mean(s5))
print("5 fold variance", np.var(s5))

print("10 fold mean", np.mean(s10))
print("10 fold variance", np.var(s10))

print("5 fold scores")
print(s5)

print("10 fold scores")
print(s10)

plt.scatter([3, 5, 6, 7, 8, 9, 10],
                [np.var(sklearn.model_selection.cross_val_score(m, X, y, cv = i)) * 100 for i in [3, 5, 6, 7, 8, 9, 10]])

