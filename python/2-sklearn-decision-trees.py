import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')

import pandas as pd
print(pd.__version__)

df = pd.read_csv('./insurance-customers-300.csv', sep=';')

y=df['group']

df.drop('group', axis='columns', inplace=True)

X = df.as_matrix()

df.describe()

# ignore this, it is just technical code
# should come from a lib, consider it to appear magically 
# http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_print = ListedColormap(['#AA8888', '#004000', '#FFFFDD'])
cmap_bold = ListedColormap(['#AA4444', '#006000', '#AAAA00'])
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#FFFFDD'])
font_size=25

def meshGrid(x_data, y_data):
    h = 1  # step size in the mesh
    x_min, x_max = x_data.min() - 1, x_data.max() + 1
    y_min, y_max = y_data.min() - 1, y_data.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return (xx,yy)
    
def plotPrediction(clf, x_data, y_data, x_label, y_label, colors, title="", mesh=True, fname=None, print=False):
    xx,yy = meshGrid(x_data, y_data)
    plt.figure(figsize=(20,10))

    if clf and mesh:
        Z = clf.predict(np.c_[yy.ravel(), xx.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    if print:
        plt.scatter(x_data, y_data, c=colors, cmap=cmap_print, s=200, marker='o', edgecolors='k')
    else:
        plt.scatter(x_data, y_data, c=colors, cmap=cmap_bold, s=80, marker='o', edgecolors='k')
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.title(title, fontsize=font_size)
    if fname:
        plt.savefig(fname)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

X_train_kmh_age = X_train[:, :2]
X_test_kmh_age = X_test[:, :2]
X_train_2_dim = X_train_kmh_age
X_test_2_dim = X_test_kmh_age

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
get_ipython().magic('time clf.fit(X_train_2_dim, y_train)')

plotPrediction(clf, X_train_2_dim[:, 1], X_train_2_dim[:, 0], 
               'Age', 'Max Speed', y_train,
                title="Train Data Max Speed vs Age with Classification")

clf.score(X_train_2_dim, y_train)

plotPrediction(clf, X_test_2_dim[:, 1], X_test_2_dim[:, 0], 
               'Age', 'Max Speed', y_test,
                title="Test Data Max Speed vs Age with Prediction")

clf.score(X_test_2_dim, y_test)

# PDF
# Needs GraphViz installed (dot)
# http://scikit-learn.org/stable/modules/tree.html
# !conda install python-graphviz -y

import graphviz 
from sklearn.tree import export_graphviz

# http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
dot_data = export_graphviz(clf, out_file=None,
                feature_names=['Spped', 'Age'],
                class_names=['red', 'green', 'yellow'],
#                 filled=True, 
                rounded=True,
                special_characters=True)

# graph = graphviz.Source(dot_data) 
# graph.render("tree") 
graphviz.Source(dot_data)

# this gives us nice pngs
# https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176
# https://graphviz.gitlab.io/download/
# !conda install -c conda-forge pydotplus -y

from sklearn.externals.six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
def plot_dt(clf):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    feature_names=['Spped', 'Age'],
                    class_names=['red', 'green', 'yellow'],
    #                 filled=True, 
                    rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())

plot_dt(clf)

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                feature_names=['Spped', 'Age'],
                class_names=['green', 'red', 'yellow'],
#                 filled=True, 
                max_depth=3,
                rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

# DecisionTreeClassifier?

clf = DecisionTreeClassifier(max_depth=6,
                              min_samples_leaf=10,
                              min_samples_split=10)
get_ipython().magic('time clf.fit(X_train_2_dim, y_train)')

plot_dt(clf)

plotPrediction(clf, X_train_2_dim[:, 1], X_train_2_dim[:, 0], 
               'Age', 'Max Speed', y_train,
                title="Train Data Max Speed vs Age with Classification")

clf.score(X_train_2_dim, y_train)

plotPrediction(clf, X_test_2_dim[:, 1], X_test_2_dim[:, 0], 
               'Age', 'Max Speed', y_test,
                title="Test Data Max Speed vs Age with Prediction")

clf.score(X_test_2_dim, y_test)



