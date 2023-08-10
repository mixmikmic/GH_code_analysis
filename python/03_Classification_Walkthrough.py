import warnings
warnings.filterwarnings('ignore', module='seaborn')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style('white')
sns.set_palette('dark')

get_ipython().magic('matplotlib inline')

from sklearn.datasets import load_iris

# import some data to play with
iris = load_iris()

# create X (features) and y (response)
X = pd.DataFrame(iris.data,
                 columns = iris.feature_names)

X.head()

y = iris.target
print(iris.target_names, np.unique(y))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Predict values for the test data
y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            annot=True, annot_kws={'fontsize':20});

from sklearn.metrics import classification_report

cr = classification_report(y_test, y_pred, 
                           target_names=iris.target_names)
print(cr)

