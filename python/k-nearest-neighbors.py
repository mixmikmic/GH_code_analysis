import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
df.head()

X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Test
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

exemple_measure = np.array([2,1,3,4,2,1,4,1,2,])
exemple_measure = exemple_measure.reshape(1,-1)

print(clf.predict(exemple_measure))

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)

