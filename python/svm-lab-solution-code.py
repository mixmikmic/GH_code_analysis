import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

from ipywidgets import *
from IPython.display import display

from sklearn.svm import SVC

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

df = pd.read_csv('../datasets/breast_cancer_wisconsin/breast_cancer.csv', na_values='?')

df.isnull().sum()

# 16 missing values in Bare_Nuclei
df['Bare_Nuclei'].hist()

# which class are the missing values?
df.ix[df['Bare_Nuclei'].isnull()]['Class'].value_counts()

# I'll drop them
df.dropna(inplace=True)
df.isnull().sum()

df['Class'].unique()

X = df.drop(['Sample_code_number', 'Class'], axis = 1)
y = df['Class'].map(lambda x: 1 if x == 4 else 0)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
Xn = ss.fit_transform(X)

#baseline => 65%
y.value_counts() / len(y)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

lin_model = SVC(kernel='linear')

scores = cross_val_score(lin_model, Xn, y, cv=5)
sm = scores.mean()
ss = scores.std()
print "Average score: {:0.3} +/- {:0.3}".format(sm, ss)

rbf_model = SVC(kernel='rbf')

scores = cross_val_score(rbf_model, Xn, y, cv=5)
sm = scores.mean()
ss = scores.std()
print "Average score: {:0.3} +/- {:0.3}".format(sm, ss)

from sklearn.metrics import classification_report

def print_cm_cr(y_true, y_pred):
    """prints the confusion matrix and the classification report"""
    confusion = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print confusion
    print
    print classification_report(y_true, y_pred)
    

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xn, y, stratify=y, test_size=0.33)
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
print_cm_cr(y_test, y_pred)

car = pd.read_csv('../datasets/car_evaluation/car.csv')

car.head(3)

car.buying.unique()

car.maint.unique()

car.lug_boot.unique()

car.safety.unique()

car.acceptability.unique()

# any na?
car.isnull().sum()

y = car.acceptability.map(lambda x: 1 if x in ['vgood','good'] else 0)

import patsy

X = patsy.dmatrix('~ buying + maint + doors + persons + lug_boot + safety -1',
                  data=car, return_type='dataframe')

ss = StandardScaler()
Xn = ss.fit_transform(X)

y.value_counts() / len(y)
# baseline is 92.2%

lin_model = SVC(kernel='linear')

scores = cross_val_score(lin_model, Xn, y, cv=5)
print scores
sm = scores.mean()
ss = scores.std()
print "Average score: {:0.3} +/- {:0.3}".format(sm, ss)

rbf_model = SVC(kernel='rbf')

scores = cross_val_score(rbf_model, Xn, y, cv=5)
print scores
sm = scores.mean()
ss = scores.std()
print "Average score: {:0.3} +/- {:0.3}".format(sm, ss)

X_train, X_test, y_train, y_test = train_test_split(Xn, y, stratify=y, test_size=0.33)
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
print_cm_cr(y_test, y_pred)

car.head()

X_train, X_test, y_train, y_test = train_test_split(Xn, y, stratify=y, test_size=0.33)
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
print_cm_cr(y_test, y_pred)

# gridsearch kNN
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

knn_params = {
    'n_neighbors':range(1,51),
    'weights':['distance','uniform']
}

knn_gs = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, verbose=1)
knn_gs.fit(Xn, y)


knn_best = knn_gs.best_estimator_
print knn_gs.best_params_
print knn_gs.best_score_

# gridsearch SVM
from sklearn.svm import SVC

svc_params = {
    'C':np.logspace(-3, 2, 10),
    'gamma':np.logspace(-5, 2, 10),
    'kernel':['linear','rbf']
}

svc_gs = GridSearchCV(SVC(), svc_params, cv=3, verbose=1)
svc_gs.fit(Xn, y)

best_svc = svc_gs.best_estimator_
print svc_gs.best_params_
print svc_gs.best_score_

from sklearn.linear_model import LogisticRegression

lr_params = {
    'penalty':['l1','l2'],
    'C':np.logspace(-4, 2, 40),
    'solver':['liblinear']
}

lr_gs = GridSearchCV(LogisticRegression(), lr_params, cv=5, verbose=1)
lr_gs.fit(Xn, y)

best_lr = lr_gs.best_estimator_
print lr_gs.best_params_
print lr_gs.best_score_

