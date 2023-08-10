import numpy as np
import io
from scipy import misc
import pydotplus as pydot

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.externals.six import StringIO 

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().magic('matplotlib inline')

# Loading the data saved from the last notebook
X_train = np.load('./_data/X_train.npy')
y_train = np.load('./_data/y_train.npy')
X_val = np.load('./_data/X_val.npy')
y_val = np.load('./_data/y_val.npy')
X_test = np.load('./_data/X_test.npy')


tree_clf = DecisionTreeClassifier(max_depth=10)

tree_clf.fit(X_train, y_train)

def show_tree(decisionTree, file_path):
    dotfile = io.StringIO()
    export_graphviz(decisionTree, out_file=dotfile)
    pydot.graph_from_dot_data(dotfile.getvalue()).write_png(file_path)
    i = misc.imread(file_path)
    plt.imshow(i)

show_tree(tree_clf, './_assets/2-1-tree-class.png')

tree_clf.score(X_val, y_val)

tree_reg = DecisionTreeRegressor(max_depth=10)
tree_reg.fit(X_train, y_train)

show_tree(tree_reg, './_assets/2-2-tree-reg.png')

tree_reg.score(X_val, y_val)



