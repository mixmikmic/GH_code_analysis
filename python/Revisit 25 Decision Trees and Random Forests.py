from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)



import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


plt.scatter(X[:,0],X[:,1], c = y,s=50,cmap='rainbow')

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X,y)

X_test,y_test = make_blobs(n_samples=30000, centers=4, random_state=0, cluster_std=2.0)

y_pred = tree.predict(X_test)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


#plt.scatter(X[:,0],X[:,1], c = y,s=50,cmap='rainbow')
plt.scatter(X_test[:,0],X_test[:,1], c = y_pred,s=50,cmap='rainbow',alpha=0.1)

tree.tree_.max_depth

import sklearn.tree as sktree



sktree.export_graphviz(tree,out_file='tree.dot')
#Use this to see tree http://www.webgraphviz.com/

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=1000)

model.fit(X,y)

y_pred = model.predict(X_test)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


#plt.scatter(X[:,0],X[:,1], c = y,s=50,cmap='rainbow')
plt.scatter(X_test[:,0],X_test[:,1], c = y_pred,s=50,cmap='rainbow',alpha=0.1)



from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_digits
digits = load_digits()

plt.imshow(digits.data[0].reshape((8,8)),cmap='gray')

digits.data[2].reshape((8,8))

digits.data.shape

digits.target

from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest = train_test_split(digits.data, digits.target, random_state=0)

rf_model = RandomForestClassifier(n_estimators=1000)

rf_model.fit(Xtrain,ytrain)

y_pred = rf_model.predict(Xtest)

from sklearn.metrics import accuracy_score

accuracy_score(ytest,y_pred)





