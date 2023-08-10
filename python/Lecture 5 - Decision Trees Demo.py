from sklearn.datasets import load_iris
from sklearn import tree
import pandas as pd
import graphviz
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from scipy.stats import randint

np.random.seed(0)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]
train = train.drop(['is_train'], axis = 1)
test = test.drop(['is_train'], axis = 1)

train_features = train[train.columns[:4]]
train_labels = pd.factorize(train['species'])[0]
test_features = test[test.columns[:4]]
test_labels = pd.factorize(test['species'])[0]

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf.fit(train_features, train_labels)

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)

graph = graphviz.Source(dot_data) 
graph

clf.score(train_features, train_labels)

clf.score(test_features, test_labels)

parameters = {"min_samples_split": [2, 10],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }
gridsearch = GridSearchCV(clf, parameters)
gridsearch.fit(train_features, train_labels)

best_tree = gridsearch.best_estimator_ 
best_tree.fit(train_features, train_labels)

dot_data = tree.export_graphviz(best_tree, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dot_data) 
graph

best_tree.score(train_features, train_labels)

best_tree.score(test_features, test_labels)

clf = RandomForestClassifier(criterion = 'entropy', n_estimators=100)
clf.fit(train_features, train_labels)

clf.score(train_features, train_labels)

clf.score(test_features, test_labels)

default_tree = tree.DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, criterion = 'entropy')

boost_clf = AdaBoostClassifier(base_estimator = default_tree)

boost_clf.fit(train_features, train_labels)

boost_clf.score(train_features, train_labels)

boost_clf.score(test_features, test_labels)

