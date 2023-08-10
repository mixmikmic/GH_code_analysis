# print all the outputs in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# imports
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
from math import log

# this allows plots to appear directly in the notebook
get_ipython().magic('matplotlib inline')

#data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
#data1 = pd.read_csv("union.csv", low_memory=False)
#data1 = pd.read_excel("union2.xlsx")
AllCleanedApps = pd.read_csv("CleanedApps.csv")
AllCleanedApps.shape
AllCleanedApps.head(5)

#Change the table values to log values
import math
def toLog(x):
    y = x
    if x == 0:
        y = 0.1
    return math.log(y)


AllCleanedApps["Downloads"] = AllCleanedApps.Downloads.apply(toLog)
AllCleanedApps["Size"] = AllCleanedApps.Size.apply(toLog)
#AllCleanedApps["Price"] = AllCleanedApps.Price.apply(toLog)
#AllCleanedApps["AvgR"] = AllCleanedApps.AvgR.apply(toLog)
AllCleanedApps["Raters"] = AllCleanedApps.Raters.apply(toLog)

import sklearn.tree as tree
import sklearn as sk
from sklearn.model_selection import train_test_split

Y=AllCleanedApps['Downloads']
X=AllCleanedApps.drop(['Downloads', 'Name','Unnamed: 0', 'Random'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=2)

X_train.shape
y_train.shape
X_train.head(3)
y_train.head(3)
X_test.shape
y_test.shape

import numpy as np
#dt = tree.DecisionTreeClassifier(max_depth=5)
dt = tree.DecisionTreeRegressor(max_depth=7)
dt.fit(X = X_train, y= y_train)

# This code will visualize a decision tree dt, trained with the attributes in X and the class labels in Y
import pydotplus
from IPython.display import Image
dt_feature_names = list(X_train.columns)
dt_target_names = np.array(y_train.unique(),dtype=np.string_) 
tree.export_graphviz(dt, out_file='tree.dot', 
    feature_names=dt_feature_names, class_names=dt_target_names,
    filled=True)  
graph = pydotplus.graph_from_dot_file('tree.dot')
Image(graph.create_png())

y_pred = dt.predict(X_test)

dt.score(X_test, y_test)

(y_test - y_pred).abs().mean()

temp = y_test.values
temp
y_pred

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
plt.figure(figsize=(15, 10))
#y_test = np.linspace(0, y_test.max, 50)
plt.scatter(y_test, y_pred)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Downloads')
plt.ylabel('Predicted Downloads')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
#plt.tight_layout()



