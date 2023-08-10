import pandas
data = pandas.read_csv("data/titanic.csv")
data.head()

y, X = data['survived'], data[['pclass','sex','age','sibsp', 'fare']].fillna(0)
X['sex'].replace(['female','male'],[0,1],inplace=True)
X['age'] = X['age'].astype(int)
y.head(), X.head()

X.columns

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=123)
len(X_train), len(y_train), len(X_test), len(y_test)

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

import os
from tempfile import mkstemp
import subprocess

from sklearn.tree.export import export_graphviz

def convert_decision_tree_to_ipython_image(clf, feature_names=None, class_names=None,
                                           image_filename=None, tmp_dir=None):
    dot_filename = mkstemp(suffix='.dot', dir=tmp_dir)[1]
    with open(dot_filename, "w") as out_file:
        export_graphviz(clf, out_file=out_file,
                        feature_names=feature_names,
                        class_names=class_names,
                        filled=True, rounded=True,
                        special_characters=True)

    import graphviz
    from IPython.display import display

    with open(dot_filename) as f:
        dot_graph = f.read()
    display(graphviz.Source(dot_graph))
    os.remove(dot_filename)

convert_decision_tree_to_ipython_image(clf, image_filename='titanic.png', feature_names=X.columns, class_names=["dead", "surv"])

clf.feature_importances_

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances(clf):
    n_features = X_train.shape[1]
    plt.barh(range(n_features), clf.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    
plot_feature_importances(clf)

from sklearn import ensemble
clf_rf = ensemble.RandomForestClassifier(n_estimators=50, random_state=800)
clf_rf.fit(X_train, y_train)
clf_rf.score(X_test, y_test)

plot_feature_importances(clf_rf)

from sklearn import ensemble
clf_gd = ensemble.GradientBoostingClassifier(random_state=800)
clf_gd.fit(X_train, y_train)
clf_gd.score(X_test, y_test)

plot_feature_importances(clf_gd)

