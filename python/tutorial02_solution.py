import numpy as np

X = np.array([[4.1,5.3],[-3.9,8.4],[6.4,-1.8]]) #YOUR CODE HERE
print(X)
print(X.shape)
print(type(X))
print(X.dtype)

x = np.array([4.1,-3.9,6.4])
print(x)
print(x.shape)

x = np.array(5.6)
print(x)
print(x.shape)

import numpy as np

X1 = np.array([[4.1,5.3],[-3.9,8.4],[6.4,-1.8]]) #YOUR CODE HERE
X2 = np.array([[2.7,3.5],[7.3,2.4],[5.0,2.8]])
X = X1 + X2
print(X1)
print(X2)
print(X)

Y1 = 4* X
Y2 = X/ 3
print(X)
print(Y1)
print(Y2)

import numpy as np

A = np.array([[4.1,5.3],[-3.9,8.4],[6.4,-1.8]]) #YOUR CODE HERE
x = np.array([2.7,3.5])
x = x[:,None]
x = np.array([[2.7],[3.5]])
y = A.dot(x)
print(A)
print(A.shape)
print(x)
print(x.shape)
print(y)
print(y.shape)

import numpy as np

A = np.array([[4.1,5.3],[-3.9,8.4],[6.4,-1.8]]) #YOUR CODE HERE
X = np.array([[2.7,3.2],[3.5,-8.2]])
Y = A.dot(X)
print(A)
print(A.shape)
print(X)
print(X.shape)
print(Y)
print(Y.shape)

import numpy as np

A = np.array([[2.7,3.5,3.2],[-8.2,5.4,-1.7]]) #YOUR CODE HERE
AT = A.T
print(AT)
print(A.shape)
print(AT.shape)

A = np.array([[2.7,3.5],[3.2,-8.2]])
Ainv = np.linalg.inv(A)
AAinv = A.dot(Ainv)
print(A)
print(A.shape)
print(Ainv)
print(Ainv.shape)
print(AAinv)
print(AAinv.shape)



