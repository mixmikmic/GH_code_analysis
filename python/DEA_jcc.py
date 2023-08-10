# use pandas to read in data
import pandas as pd

df_X = pd.read_csv('../data/train_X.csv')
df_y = pd.read_csv('../data/train_y.csv')
df = pd.merge(df_y, df_X, on='id')

# -- convert functional to 2, non-functional to 0, functional-needs-repaird to 1

df.loc[:, 'status'] = df.status_group
df.loc[df.status.str.startswith('functional needs'), 'status'] = 1
df.loc[df.status_group.str.startswith('non'), 'status'] = 0
df.loc[~df.status.str.startswith('functional').isnull(), 'status'] = 2
df.status = df.status.astype(int)

# -- show df.corr()
df.corr()

# import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import patsy

df.head()

cols = list(df.columns.values)

# keep columns
cols_keep = []
for c in df:
    if df[c].dtype in [int, float]:
        cols_keep.append(c)
    elif df[c].dtype == object:
        if df[c].nunique() < 20:
            cols_keep.append(c)

# remove the labels
for to_remove in ['id', 'status', 'status_group']:
    cols_keep.remove(to_remove)

# convert df to X, y by patsy
r_formula = 'status ~' + ' + '.join(cols_keep)
df_y, df_X = patsy.dmatrices(r_formula, df, return_type='dataframe')

cols_X = df_X.columns
X = df_X.values
y = df_y.values

y

X

def split_n_fit(model, X, y):
    """ given model, X, y, print score of the fit on test """
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), random_state=42)
    model.fit(X_train, y_train)
    print
    print '{}'.format(model).split('(')[0]
    print model.score(X_test, y_test)


for model in [LogisticRegression(), DecisionTreeClassifier(),
              KNeighborsClassifier(), GaussianNB(), RandomForestClassifier()]:
    split_n_fit(model, X, y)

# it looks RandomForest does us better, let's try a new parameter

model = RandomForestClassifier(n_estimators=200)
split_n_fit(model, X, y)

# it looks RandomForest does us better, let's try a new parameter

model = RandomForestClassifier(n_estimators=300)
split_n_fit(model, X, y)

from sklearn.cross_validation import KFold

model = RandomForestClassifier(n_estimators=200)
y = y.ravel()
for train_index, test_index in KFold(len(y), n_folds=5):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    model.fit(X_train, y_train)
    print model.score(X_test, y_test)



from sklearn.metrics import confusion_matrix
# get y_pred and y_actual
y_pred = model.predict(X_test)
y_actual = y_test

# convert y's to binary

# y_pred_bin = ~y_pred.astype(bool)  
# y_actual_bin = ~y_actual.astype(bool)  #

# we want True means not-functional or need-repair (where the govern. needs to send people) , False: functional
# In our data, 2: functional 1: need repair 0: non-functional
y_pred_bin = (y_pred < 2)
y_actual_bin = (y_actual < 2)

conf = confusion_matrix(y_actual_bin, y_pred_bin)
conf

TP = conf[0, 0]
FN = conf[0, 1]
FP = conf[1, 0]
TN = conf[1, 1]
recall = TP * 1./ (TP + FN)
print "recall", recall
precision = TP * 1./ (TP + FP)
print "precision", precision

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = importances.argsort()[::-1]
print "Features ranked by importance:"
for i, (feature, importance) in enumerate(zip(cols_X[indices], importances[indices])):
    print i, feature, importance

# just plot the first 20
n_first = 20
plt.figure(figsize=[15, 10])
plt.bar(range(n_first), importances[indices[0:20]], color='g',
        yerr=std[indices[0:20]], align='center')
plt.xticks(range(n_first), cols_X[indices[0:20]], rotation=60)
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix

get_ipython().magic('pinfo confusion_matrix')



