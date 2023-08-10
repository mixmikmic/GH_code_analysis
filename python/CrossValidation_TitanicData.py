import os, sys
from collections import Counter
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.naive_bayes import GaussianNB

np.random.seed(18937)

datasource = "datasets/titanic.csv"
print(os.path.exists(datasource))

df = pd.read_csv(datasource).sample(frac = 1).reset_index()
df.head()

df.describe()

m = GaussianNB()

X = np.array(df.iloc[:, :-1]) # give me everything except for the last column (label)
y = np.array(df["survived"]) # just give me the label column (survived)

sklearn.model_selection.cross_val_score(m, X, y, cv = 20)

def cross_val_score(model, X, y, cv):
    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)
    
    for i in range(cv):
        X_train = np.concatenate([X_folds[j] for j in range(cv) if j != i])
        X_test = X_folds[i]
        
        y_train = np.concatenate([y_folds[j] for j in range(cv) if j != i])
        y_test = y_folds[i]
        
        model.fit(X_train, y_train)
        yield model.score(X_test, y_test)

for i, score in enumerate(cross_val_score(m, X, y, cv = 20)):
    print("Score for CV #", i, "is ", score)



