import numpy as np
import scipy

from sklearn import datasets
from sklearn.model_selection  import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

data = datasets.load_digits()

X_data = data.images   # load X_data
y_data = data.target   # load y_data

X_data = X_data.reshape(X_data.shape[0], X_data.shape[1] * X_data.shape[2])    # flatten X_data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 7)    # split data into train & test set

clf = MLPClassifier(hidden_layer_sizes = (5, 5), max_iter = 500)   # create a MLP with two hidden layers with five neurons
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_pred, y_test))

clf1 = MLPClassifier(hidden_layer_sizes = (5, 5), max_iter = 500)   # create a MLP with two hidden layers with five neurons
clf2 = MLPClassifier(hidden_layer_sizes = (5, 5), max_iter = 500)   # create a MLP with two hidden layers with five neurons
clf3 = MLPClassifier(hidden_layer_sizes = (5, 5), max_iter = 500)   # create a MLP with two hidden layers with five neurons

ensemble_clf = VotingClassifier(estimators = [('clf1', clf1), ('clf2', clf2), ('clf3', clf3)], voting = 'hard')

ensemble_clf.fit(X_train, y_train)

y_pred = ensemble_clf.predict(X_test)

print(accuracy_score(y_pred, y_test))

ensemble_clf = VotingClassifier(estimators = [('clf1', clf1), ('clf2', clf2), ('clf3', clf3)], voting = 'soft')

ensemble_clf.fit(X_train, y_train)

y_pred = ensemble_clf.predict(X_test)

print(accuracy_score(y_pred, y_test))

