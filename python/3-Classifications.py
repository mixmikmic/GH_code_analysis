from sklearn import tree

# features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
# labels = ["apple", "apple", "orange", "orange"]

# smooth : 1, bumpy : 0
# apple : 0, orange : 1

#features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# Checking with more features
# labels = [0, 0, 1, 1]
features = [[140, 1], [130, 1], [150, 0], [170, 0],[180, 1], [120, 1], [142, 0], [150, 0]]
labels = [0, 1, 1, 1, 1, 1, 1, 0]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

print(classifier.predict([[145, 0]]))

from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

# Instead of reading from our CSV file, Scikit-Learn provides us with the Iris dataset
iris = load_iris()
test_idx = [0, 50, 100]

# Training data
training_target = np.delete(iris.target, test_idx)
training_data = np.delete(iris.data, test_idx, axis = 0)

# Testing data
testing_target = iris.target[test_idx]
testing_data = iris.data[test_idx]

classifier = tree.DecisionTreeClassifier()
classifier.fit(training_data, training_target)

print(classifier.predict(testing_data[:1]))
print(testing_data[0], testing_target[0])
print(iris.feature_names, iris.target_names)

import csv
import numpy as np
from sklearn import tree

# Read data to lists and identify features

# Plot and understand your features

# Make a Scikit Learn classifier

# Fit the model using your classifier

# Determine level of correctness



