from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

type(iris)

isinstance(iris, dict)

iris.keys()

iris.feature_names

iris.target_names

iris['target']

type(iris['target'])

for i in range(len(iris.target)):
    if i < 5:
        print('Example {}: label {}, features {}'.format(i, iris.target[i], iris.data[i]))

import numpy as np

test_idx = [0, 50, 100]  # these are the rows to be removed from the training data

# remove the same rows from the actual data
# Note: without axis=0, returns just a  list, not a list of lists
# ie we want this:
# [[ 4.9,  3. ,  1.4,  0.2],
#       [ 4.7,  3.2,  1.3,  0.2],
#       [ 4.6,  3.1,  1.5,  0.2], …]
# and not this:
# [4.9,  3. ,  1.4,  0.2,  4.7,  3.2,  1.3,  0.2, 4.6,  3.1,  1.5,  0.2, …]
train_data = np.delete(iris.data, test_idx, axis=0)

# np.delete() remove the above 3 indices from array iris.target
# Note: here the axis= arg doesn't matter, as only a 1 interger per item in list
train_target = np.delete(iris.target, test_idx)

# See how rows have been rm'd
len(iris.target)

len(train_data)  # the three taken out

len(train_target)

test_target = iris.target[test_idx]

test_target  # only three

test_data = iris.data[test_idx]

test_data

# Note: on numpy array
l = [1, 4, 5,6,8, 999, 44, 6, 7, 10]
a = np.array(l)

# now we can pull out items with a list of indices/rows
idx = [0, 4, 6]
a[idx]

# train model
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# make prediction
clf.predict(test_data)

# matches input labels?
clf.predict(test_data) == test_target

from sklearn.externals.six import StringIO
import pydotplus # note installed pydotplus for Py3 compatibility

dot_data = StringIO()

tree.export_graphviz(clf, 
                     out_file=dot_data, 
                     feature_names=iris.feature_names, 
                     class_names=iris.target_names, 
                     filled=True, 
                     rounded=True, 
                     impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# graphviz installed on mac with `brew install graphviz`
graph.write_pdf('iris.pdf')

# open -a preview ~/ipython/tensorflow/iris.pdf 

# now check the rows withheld for testing
# check against rules in graphic tree
test_data[0], test_target[0]  # we know is a setosa

iris.feature_names, iris.target_names

test_data[1], test_target[1]  # we know is a versicolor

test_data[2], test_target[2]  # we know is a virginica

# all test to true!

