import numpy as np
np.random.seed(1)

from sklearn.datasets import load_iris
iris = load_iris()

iris.keys()

iris.DESCR

print(iris.feature_names)
print(len(iris.feature_names))
print()
print(iris.target_names)
print(len(iris.target_names))

print(len(iris.data))
print(type(iris.data))
iris.data

print(len(iris.target))
print(type(iris.target))
iris.target

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    train_size=0.75, test_size=0.25)

from sklearn import tree

dt_classifier = tree.DecisionTreeClassifier(criterion='gini',  # or 'entropy' for information gain
                       splitter='best',  # or 'random' for random best split
                       max_depth=None,  # how deep tree nodes can go
                       min_samples_split=2,  # samples needed to split node
                       min_samples_leaf=1,  # samples needed for a leaf
                       min_weight_fraction_leaf=0.0,  # weight of samples needed for a node
                       max_features=None,  # number of features to look for when splitting
                       max_leaf_nodes=None,  # max nodes
                       min_impurity_split=1e-07)  # early stopping

model = dt_classifier.fit(X_train, y_train)
print(model.score(X_test, y_test))

print(model.decision_path(X_test)) # Return a node indicator matrix where non zero elements indicates that the samples goes through the nodes.

from sklearn.model_selection import GridSearchCV

param_grid = {'min_samples_split': range(2,10),
              'min_samples_leaf': range(1,10)}

model_c = GridSearchCV(tree.DecisionTreeClassifier(), param_grid)
model_c.fit(X_train, y_train)

best_index = np.argmax(model_c.cv_results_["mean_test_score"])

print(model_c.cv_results_["params"][best_index])
print(max(model_c.cv_results_["mean_test_score"]))
print(model_c.score(X_test, y_test))

from sklearn.datasets import load_boston

boston = load_boston()

boston.keys()

boston.DESCR

print(boston.feature_names)
print()
print(type(boston.feature_names))
print()
print(len(boston.feature_names))

print(boston.data)
print()
print(type(boston.data))
print()
print(len(boston.data))

print(boston.target)
print()
print(type(boston.target))
print()
print(len(boston.target))

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target,
                                                    train_size=0.75, test_size=0.25)

print(len(X_train), len(y_train))
print()
print(len(X_test), len(y_test))

from sklearn import tree

dt_reg = tree.DecisionTreeRegressor(criterion='mse',  # how to measure fit
                                    splitter='best',  # or 'random' for random best split
                                    max_depth=None,  # how deep tree nodes can go
                                    min_samples_split=2,  # samples needed to split node
                                    min_samples_leaf=1,  # samples needed for a leaf
                                    min_weight_fraction_leaf=0.0,  # weight of samples needed for a node
                                    max_features=None,  # number of features to look for when splitting
                                    max_leaf_nodes=None,  # max nodes
                                    min_impurity_split=1e-07)  # early stopping

model = dt_reg.fit(X_train, y_train)
print(model.score(X_test, y_test))

print(model.decision_path(X_train))

param_grid = {'min_samples_split': range(2,10),
              'min_samples_leaf': range(1,10)}

model_r = GridSearchCV(tree.DecisionTreeRegressor(), param_grid)
model_r.fit(X_train, y_train)

best_index = np.argmax(model_r.cv_results_["mean_test_score"])

print(model_r.cv_results_["params"][best_index])
print(max(model_r.cv_results_["mean_test_score"]))
print(model_r.score(X_test, y_test))

