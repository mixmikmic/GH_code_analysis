import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

N = 1000
blue_X = np.random.rand(N,2) @ np.array([[1,1],[-1,1]]) + np.array([1,1]).reshape(1,2)
red_X = np.random.rand(N,2) @ np.array([[1,1],[-1,1]]) + np.array([2,2]).reshape(1,2)

plt.scatter(blue_X[:,0],blue_X[:,1],c='b',lw=0,alpha=0.2);
plt.scatter(red_X[:,0],red_X[:,1],c='r',lw=0,alpha=0.2);
plt.axis('equal');

X = np.vstack([blue_X,red_X])
y = np.concatenate([np.zeros(N),np.ones(N)]) # Blue is 0 and Red is 1

from sklearn.linear_model import LogisticRegression as LR

# Instantiate the model
reg = LR()

# Fit the model on the fake data
reg.fit(X,y)

X_test = np.array([[a,b] for a in np.linspace(0,3,50) for b in np.linspace(1,4,50)])
y_test = reg.predict(X_test)

plt.scatter(blue_X[:,0],blue_X[:,1],c='b',lw=0,alpha=0.2);
plt.scatter(red_X[:,0],red_X[:,1],c='r',lw=0,alpha=0.2);

plt.scatter(X_test[:,0],X_test[:,1],c=y_test,lw=0,alpha=0.2)
plt.axis('equal');

# This is our weight matrix W
reg.coef_

# This is our bias b
reg.intercept_

from sklearn.neural_network import MLPClassifier as mlp

nn = mlp(hidden_layer_sizes=(1,),activation='logistic',max_iter=2000,alpha=0.01)

nn.fit(X,y)

y_nn_test = nn.predict(X_test)

plt.scatter(blue_X[:,0],blue_X[:,1],c='b',lw=0,alpha=0.2);
plt.scatter(red_X[:,0],red_X[:,1],c='r',lw=0,alpha=0.2);

plt.scatter(X_test[:,0],X_test[:,1],c=y_nn_test,lw=0,alpha=0.2)
plt.axis('equal');

nn.coefs_

from sklearn.datasets import load_digits

digits = load_digits()

X = digits.data

y = digits.target

X.shape

plt.imshow(X[0,:].reshape(8,8),cmap='binary',interpolation='none');

from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(X,y,test_size=0.2)

nn_digits = mlp(hidden_layer_sizes=(10,10,10),max_iter=1000)

nn_digits.fit(X_train,y_train)

nn_digits.score(X_test,y_test)

from sklearn.neighbors import KNeighborsClassifier as knn

knn_clf = knn(n_neighbors=7)

knn_clf.fit(X_train,y_train)

knn_clf.score(X_test,y_test)

