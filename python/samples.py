import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn import datasets
from pydotplus import graph_from_dot_data
from IPython.display import Image
 
# Load data
iris = datasets.load_iris()
features = iris.data
categories = iris.target
 
# Cross-Validation setting
X_train, X_test, y_train, y_test = train_test_split(features, categories, test_size=0.2, random_state=42)
cv_sets = ShuffleSplit(X_train.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
params = {'max_depth': np.arange(2,11), 'min_samples_leaf': np.array([5])}
 
# Learning
def performance_metric(y_true, y_predict):
  score = f1_score(y_true, y_predict, average='micro')
  return score
 
classifier = DecisionTreeClassifier()
scoring_fnc = make_scorer(performance_metric)
grid = GridSearchCV(classifier, params, cv=cv_sets, scoring=scoring_fnc)
best_clf = grid.fit(X_train, y_train)
 
# Plot decision tree
dot_data = export_graphviz(best_clf.best_estimator_, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graph_from_dot_data(dot_data)  
Image(graph.create_png())

from MeCab import Tagger
from pyknp import Juman
 
target_text = u"すもももももももものうち"
 
m = Tagger("-Owakati")
print("***** Mecab *****")
print(m.parse(target_text))
 
juman = Juman()
result = juman.analysis(target_text)
print("***** Juman *****")
print(' '.join([mrph.midasi for mrph in result.mrph_list()]))

import numpy as np
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist

## 1. Load data
X_train, y_train, X_test, y_test = mnist.load_data(one_hot=True)

import tfgraphviz as tfg

## 2. Build Model
tf.reset_default_graph()
net = tflearn.input_data([None, X_train.shape[1]])
net = tflearn.fully_connected(net, 128, activation='ReLU')
net = tflearn.fully_connected(net, 32, activation='ReLU')
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=3)

## 3. Traning
model.fit(X_train, y_train, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=20)

## 4. Testing
predictions = np.array(model.predict(X_test)).argmax(axis=1)
actual = y_test.argmax(axis=1)
test_accuracy = np.mean(predictions == actual, axis=0)
print("Test accuracy: ", test_accuracy)

import seaborn as sns
get_ipython().magic('matplotlib inline')

sns_iris = sns.load_dataset("iris")
sns.jointplot('sepal_width', 'petal_length', data=sns_iris)

tf.reset_default_graph()

# Creating input and correct result data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Build network
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Output graph with tfgraphviz
tfg.board(tf.get_default_graph())



