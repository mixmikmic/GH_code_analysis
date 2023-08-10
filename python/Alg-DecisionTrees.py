# Simple example

from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

# Simple predict

print(clf.predict([[2., 2.]]))
print(clf.predict_proba([[2., 2.]]))

# Complex example

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# save to plot
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
    
# if you install Graphviz in your system, you can plot the tree.
# dot -Tpdf iris.dot -o iris.pdf

# See the result here:
# http://scikit-learn.org/stable/modules/tree.html

# result

print(clf.predict(iris.data[:1, :]))
print(clf.predict_proba(iris.data[:1, :]))

