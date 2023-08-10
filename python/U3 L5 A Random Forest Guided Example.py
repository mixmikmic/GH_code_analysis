import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

y2015 = pd.read_csv('LoanStats3d.csv', skipinitialspace=True, header=1)

y2015.head()

# from sklearn import ensemble
# from sklearn.model_selection import cross_val_score

# rfc = ensemble.RandomForestClassifier()
# X = y2015.drop('loan_status', 1)
# Y = y2015['loan_status']
# X = pd.get_dummies(X)

# cross_val_score(rfc, X, Y, cv=5)

# Doing this by itself will crash the kernel.

categorical = y2015.select_dtypes(include=['object'])
for i in categorical:
    column = categorical[i]
    print(i)
    print(column.nunique())

# Convert ID and interest rate to numeric.
y2015['id'] = pd.to_numeric(y2015['id'], errors='coerce')
y2015['int_rate'] = pd.to_numeric(y2015['int_rate'].str.strip('%'), errors='coerce')

# Drop other columns with many unique variables.
y2015.drop(['url', 'emp_title', 'zip_code', 'earliest_cr_line', 
           'revol_util', 'sub_grade', 'addr_state', 'desc'], 1, inplace=True)

y2015.tail()

# Remove two summary rows at the end that don't actually contain data.
y2015 = y2015[:-2]

pd.get_dummies(y2015)

from sklearn import ensemble
from sklearn.model_selection import cross_val_score

rfc = ensemble.RandomForestClassifier()
X = y2015.drop('loan_status', 1)
Y = y2015['loan_status']
X = pd.get_dummies(X)
X = X.dropna(axis=1)

cross_val_score(rfc, X, Y, cv=10)

# Data is way too large, just use PCA to get the 
# new components that will explain most of the variance. 
from sklearn import preprocessing
from sklearn.decomposition import PCA

X_scaled = pd.DataFrame(preprocessing.scale(X), columns = X.columns)

# PCA
pca = PCA(n_components = 8)
pca.fit_transform(X_scaled)

PCA_X = pca.fit_transform(X_scaled)

cross_val_score(rfc, PCA_X, Y, cv=10)



