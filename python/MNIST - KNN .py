
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# Read train and test data into Python

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

# Classification Labels

Label = train.pop("label")

# Split train data into in train and in test

in_train = train[:38000]
in_test = train[38000:42000]

in_train_label = Label[:38000]
in_test_label = Label[38000:42000]

# K-NN with 1 neighbour

knn_1 = KNeighborsClassifier(n_neighbors = 5)
knn_1.fit(in_train, in_train_label)

# Classifying the test set

Predictions = knn_1.predict(in_test)

# Accuracy

acc = np.mean(Predictions == in_test_label) * 100

acc



