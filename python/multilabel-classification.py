get_ipython().magic("config InlineBackend.figure_format='retina'")
get_ipython().magic('matplotlib inline')

import numpy as np
np.random.seed(3)
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import train_test_split

yeast = fetch_mldata('yeast')
X = yeast.data
y = yeast.target.T.toarray()
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

et = ExtraTreesClassifier()
et.fit(X_train, y_train)

rf.score(X_test, y_test)

et.score(X_test, y_test)

print(et.predict_proba(X_test[:2,:])[0])
print(y_test[:2])
y_pred = et.predict(X_test)
average_precision_score(y_test, y_pred)

from sklearn.metrics import average_precision_score

y = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
y_ = np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]])

average_precision_score(y, y_)

from sklearn.datasets import fetch_20newsgroups
categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,)

data_test.target[:10]

fetch_mldata("wine quality").data.shape

