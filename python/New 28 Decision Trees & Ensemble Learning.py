from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


plt.scatter(X[:,0],X[:,1], c = y,s=50,cmap='rainbow')

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5).fit(X,y)

X_test,y_test = make_blobs(n_samples=30000, centers=4, random_state=0, cluster_std=2.0)

y_pred = tree.predict(X_test)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#plt.scatter(X[:,0],X[:,1], c = y,s=50,cmap='rainbow')
plt.scatter(X_test[:,0],X_test[:,1], c = y_pred,s=50,cmap='rainbow',alpha=0.1)

import sklearn.tree as sktree

sktree.export_graphviz(tree,out_file='tree.dot')
#http://www.webgraphviz.com/

X[0]

y

tree.tree_.max_depth

#Decison trees get into the case of overfitting

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home='.')

#image is of 28 X 28
mnist.data.shape

from sklearn.model_selection import train_test_split

trainX,testX,trainY,testY = train_test_split(mnist.data,mnist.target)

tree = DecisionTreeClassifier().fit(trainX,trainY)

from sklearn.metrics import accuracy_score
pred = tree.predict(testX)
print (accuracy_score(testY,pred))

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
#n_estimators - num of decision trees used to create a forest
random_forest = RandomForestClassifier(n_estimators=40)

random_forest.fit(trainX,trainY)

random_forest.base_estimator_

pred = random_forest.predict(testX)
print (accuracy_score(testY,pred))

#Random Forest is useful in finding feature importances
random_forest.feature_importances_.shape

adaboost = AdaBoostClassifier(n_estimators=40)

adaboost.fit(trainX,trainY)

pred = adaboost.predict(testX)
print (accuracy_score(testY,pred))

from sklearn.linear_model import SGDClassifier
adaboost = AdaBoostClassifier(SGDClassifier(loss='hinge'),n_estimators=40,algorithm='SAMME')

adaboost.fit(trainX,trainY)

pred = adaboost.predict(testX)
print (accuracy_score(testY,pred))

random_forest.feature_importances_



