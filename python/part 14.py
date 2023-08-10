from IPython.display import Math

import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors
import matplotlib.pyplot as plt
from matplotlib import style

get_ipython().magic('matplotlib inline')
style.use('fivethirtyeight')

df = pd.read_csv('dataset/breast-cancer-wisconsin.data')

# Clean up the dataset as described in names point 8
df.replace('?', -99999, inplace=True)

# drop tables that are useless
df.drop(['id'], 1, inplace=True)

# Create the features, everything except the class (and id) column
X = np.array(df.drop(['class'], 1))

# Create the label column
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_jobs=4)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
accuracy

example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
example_measures = example_measures.reshape(1, -1)

prediction = clf.predict(example_measures)
prediction

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
prediction

