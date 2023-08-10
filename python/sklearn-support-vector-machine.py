import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# replace unkown data with outliers
df.replace('?',-99999, inplace=True)
# irrelevent feature
df.drop(['id'], 1, inplace=True)

# feature data
X = np.array(df.drop(['class'], 1))
# class / label data
y = np.array(df['class'])

# separate training and testing chunks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define classifier
clf = svm.SVC()

# train classifier
clf.fit(X_train, y_train)

# test
accuracy = clf.score(X_test, y_test)
print('accuracy:', accuracy)
# about 96% accuracy without any tweaks
# If you want to save the classier you'd pickle it

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1], [1,0,6,1,5,1,2,4,2]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print('example class outputs: ', prediction)

