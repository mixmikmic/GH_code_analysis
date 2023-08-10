import os, sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.base import clone

np.set_printoptions(suppress = True) # no scientific notation

datasource = "datasets/titanic.csv"
print(os.path.exists(datasource))

df = pd.read_csv(datasource).sample(frac = 1).reset_index(drop = True)
df.head()

X = np.array(df.iloc[:, :-1])

y = np.array(df["survived"])

print(X.shape)

print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

selector = SelectKBest(chi2, k = 5) # top 5 features

selector.fit(X, y)

print("χ² statistic:", selector.scores_)

print("Column indices:", selector.get_support(True))

selectedColumnNames = np.array(df.columns[selector.get_support(True)])
print("Column names:", selectedColumnNames)

X_train_selected = selector.transform(X_train)
print(X_train_selected.shape)

X_test_selected = selector.transform(X_test)
print(X_test_selected.shape)

# This is the selected score
model = GaussianNB()
model.fit(X_train_selected, y_train)
selectedFeaturesScore = model.score(X_test_selected, y_test)
print("Selected features score:", selectedFeaturesScore)

# This is the score without feature selection
model = GaussianNB()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Score without feature selection:", score)

chi2_sklearn, pvalue_sklearn = chi2(X_train, y_train)
print(pvalue_sklearn)

chi2_sklearn, pvalue_sklearn = chi2(X_train_selected, y_train)
print(pvalue_sklearn)

def mutual_info_session(X_train, y_train):
    selector = SelectKBest(mutual_info_classif, k = 3)
    selector.fit(X_train, y_train)
    print(selector.get_support(True))
    
    model = GaussianNB()
    model.fit(selector.transform(X_train), y_train)
    return model.score(selector.transform(X_test), y_test)

mutual_info_session(X_train, y_train)

class ForwardSelector(object):
    def __init__(self, estimator):
        self.estimator = estimator
        
    def fit(self, X, y, k):
        selected = np.zeros(X.shape[1]).astype(bool) # holds indicators of whether each feature is selected
        
        score = lambda X_features: clone(self.estimator).fit(X_features, y).score(X_features, y)
        # fit and score model based on some subset of features
        
        selected_indices = lambda: list(np.flatnonzero(selected))
        
        while np.sum(selected) < k: # keep looping until k features are selected
            rest_indices = list(np.flatnonzero(~selected)) # indices to unselected columns
            scores = list()
            
            for i in rest_indices:
                feature_subset = selected_indices() + [i]
                s = score(X[:, feature_subset])
                scores.append(s)
            idx_to_add = rest_indices[np.argmax(scores)]
            selected[idx_to_add] = True
        self.selected = selected.copy()
        return self
    
    def transform(self, X):
        return X[:, self.selected]
    
    def get_support(self, indices = False):
        return np.flatnonzero(self.selected) if indices else self.selected

def forward_selection_session(X_train, y_train):
    model = GaussianNB()
    selector = ForwardSelector(model)
    selector.fit(X_train, y_train, 5)
    print(selector.get_support(True))
    
    model = GaussianNB()
    model.fit(selector.transform(X_train), y_train)
    return model.score(selector.transform(X_test), y_test)

forward_selection_session(X_train, y_train)

def rfe_session(X_train, y_train):
    from sklearn.svm import SVC
    model = SVC(kernel = "linear")
    selector = RFE(model, 5)
    selector.fit(X_train, y_train)
    print(selector.get_support(True))
    
    model = SVC(kernel = "linear")
    model.fit(selector.transform(X_train), y_train)
    return model.score(selector.transform(X_test), y_test)

rfe_session(X_train, y_train)

