# Base
import numpy as np
import pandas as pd
import math

# Machine Learning
from sklearn import datasets, metrics, cross_validation, neural_network

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error

from sknn.mlp import Regressor, Classifier, Layer
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
# from sknn.mlp import Regressor, Layer

# Import Pre-Processed Wav File Data Set
wavData = pd.read_csv('feature_quant.csv')

wavData[0:5]

# Remove Empty Rows
wavData = wavData[-np.isnan(wavData['mean'])]

len(wavData)

from collections import Counter
Counter(wavData['class'])

feat = list(wavData.columns)
feat.remove('class')
feat.remove('Unnamed: 0')
feat

X_train, X_test, y_train, y_test = train_test_split(wavData.loc[:,feat], wavData.loc[:,'class'],                                                     test_size=0.3, random_state=0)

sc = StandardScaler()
sc=sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=8)
tree.fit(X_train_std, y_train)
y_predict = tree.predict(X_test_std)

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test,y_predict))
from sklearn.metrics import confusion_matrix
confmat=confusion_matrix(y_true=y_test, y_pred=y_predict)
print(confmat)

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
plotly.offline.init_notebook_mode()
init_notebook_mode()

py.sign_in('hm7zg', 'b8nrsfeca7')

data = [
    go.Heatmap(
        z=confmat
    )
]
plot_url = py.plot(data, filename='basic-heatmap')

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
y_predict = tree.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test,y_predict))
from sklearn.metrics import confusion_matrix
confmat=confusion_matrix(y_true=y_test, y_pred=y_predict)
print(confmat)















forest = RandomForestClassifier(criterion='entropy',n_estimators=10, class_weight="balanced",
                               random_state=1,
                               n_jobs=2)
forest.fit(X_train, y_train)

y_predict = forest.predict(X_test)

y_predict[0:5]

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test,y_predict))
from sklearn.metrics import confusion_matrix
confmat=confusion_matrix(y_true=y_test, y_pred=y_predict)
print(confmat)

# Create new response variable
wavData['gunflag'] = 0
wavData.loc[wavData['class'] == 'gun_shot','gunflag'] = 1

feat = list(wavData.columns)
feat.remove('class')
feat.remove('Unnamed: 0')
feat.remove('gunflag')
feat

X_train, X_test, y_train, y_test = train_test_split(wavData.loc[:,feat], wavData.loc[:,'gunflag'],                                                     test_size=0.35, random_state=0)

forest = RandomForestClassifier(criterion='entropy',n_estimators=10,
                               random_state=1,
                               n_jobs=2)
forest.fit(X_train, y_train)

y_predict = forest.predict(X_test)

print('Accuracy: %.2f' % accuracy_score(y_test,y_predict))
from sklearn.metrics import confusion_matrix
confmat=confusion_matrix(y_true=y_test, y_pred=y_predict)
print(confmat)

y_test.sum()

# Set up loop to get Accuracy for each class as 1-vs-All
def runRFonevsall(var, wavData):
    # Create new response variable
    wavData[var] = 0
    wavData.loc[wavData['class'] == var,var] = 1
    feat = list(wavData.columns)
    #print(feat)
    feat.remove('class')
    feat.remove('Unnamed: 0')
    feat.remove(var)
    #print(feat)
    X_train, X_test, y_train, y_test = train_test_split(wavData.loc[:,feat], wavData.loc[:,var],                                                         test_size=0.35, random_state=0)
    print(var)
    forest = RandomForestClassifier(criterion='entropy',n_estimators=10,
                                   random_state=1,
                                   n_jobs=2)
    forest.fit(X_train, y_train)
    y_predict = forest.predict(X_test)
    print('Accuracy: %.2f' % accuracy_score(y_test,y_predict))
    print('Precision: %.2f' % precision_score(y_test,y_predict))
    print('Recall: %.2f' % recall_score(y_test,y_predict))
    print('F1: %.2f' % f1_score(y_test,y_predict))
    from sklearn.metrics import confusion_matrix
    confmat=confusion_matrix(y_true=y_test, y_pred=y_predict, labels = [1,0])
    print(confmat)
    y_test.sum()
    return {var:(accuracy_score(y_test,y_predict),precision_score(y_test,y_predict), recall_score(y_test,y_predict), f1_score(y_test,y_predict))}

runRFonevsall('gun_shot', wavData)



classes = set(wavData['class']); classes

results = {}
[runRFonevsall(var, wavData) for var in classes]



