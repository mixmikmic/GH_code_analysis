get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
mnist = fetch_mldata("MNIST original")
# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
print mnist.target.shape,mnist.data.shape
X_train, X_test = X[:65000], X[65000:]
y_train, y_test = y[:65000], y[65000:]

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
y_hat=mlp.predict(X_train)
print 'Training accuracy using accuracy_score function',accuracy_score(y_train,y_hat)
y_hat=mlp.predict(X_test)
print 'Training accuracy using accuracy_score function',accuracy_score(y_test,y_hat)

import numpy as np
k=y_test!=y_hat
print k

itemindex = np.where(k==True)

print itemindex[0]
print itemindex[0].shape

print len(itemindex[0])

random_index_match=1423
random_index_no_match=1422

print X_test[random_index_match].shape
match_image=np.reshape(X_test[random_index_match],(28,28))
plt.imshow(match_image,cmap='gray')

print y_test[random_index_match]

print X_test[random_index_no_match].shape
no_match_image=np.reshape(X_test[random_index_no_match],(28,28))
plt.imshow(no_match_image,cmap='gray')

print y_test[random_index_no_match],y_hat[random_index_no_match]

y=[0,1,2,3,7]
print y[:3],y[3:],y[:-1]



