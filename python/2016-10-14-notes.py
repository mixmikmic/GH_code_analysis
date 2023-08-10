import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')

from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

diabetes.keys()

X = diabetes.data
y = diabetes.target

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X,y)

reg.score(X,y)

reg.coef_

reg.intercept_

Xm = np.matrix( np.hstack((np.ones(442).reshape(442,1),X)) ) # Array X as a NumPy matrix
ym = np.matrix(y).T # Array y as a NumPy matrix

A = (Xm.T * Xm)**(-1) * Xm.T * ym # Compute the coefficients

A

R2 = 1 - np.sum((y - reg.predict(X))**2)/np.sum((y - np.mean(y))**2)

R2

reg.score(X,y)

Xnew = np.hstack((X,X**2)) # New feature array has 20 features

Xnew.shape

reg.fit(Xnew,y)

reg.score(Xnew,y)

num_samples = 300
X_train = Xnew[:num_samples,:]
X_test = Xnew[num_samples:,:]
y_train = y[:num_samples]
y_test = y[num_samples:]

reg.fit(X_train,y_train)

reg.score(X_test,y_test)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xnew,y,test_size=0.3)

reg.fit(X_train,y_train)

reg.score(X_test,y_test)

from sklearn.datasets import load_digits

digits = load_digits()

digits.keys()

images = digits.images

images.shape

plt.imshow(images[1001,:,:],cmap='binary')

images[1001,:,:]

from sklearn.neighbors import KNeighborsClassifier as KNN

clf = KNN(n_neighbors = 10)

X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.3)

clf.fit(X_train,y_train)

clf.score(X_test,y_test)

plt.imshow(X_test[10,:].reshape(8,8),cmap='binary')

clf.predict(X_test[10,:].reshape(1,64))

