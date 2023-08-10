import sklearn.tree as tr
from sklearn.datasets import load_iris
import os
os.chdir('C:\\Users\\Harrison\\Documents\\GitHub\\ML-Notes')
from VisualFuncs import VDR
get_ipython().run_line_magic('matplotlib', 'inline')


iris = load_iris()
X = iris.data
y = iris.target

tree = tr.DecisionTreeClassifier(max_depth =3)
tree.fit(X[:,[2,3]],y)

VDR( X[:,[2,3]], y, tree)


