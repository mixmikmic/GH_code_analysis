import numpy as np

X = 10*(2 * np.random.rand(100, 1) - 1)
y = 2*X + 3 + 15*np.random.rand(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
m=100

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(X, y)
plt.show()

n_epochs = 200
eta = 0.0001
batch_size = 50

theta = np.random.randn(2, 1)
plt.scatter(X, y)

for epoch in range(n_epochs):
    for i in range(m):
        
        # Sample Mini batch
        shuffle_index = np.random.permutation(m)
        xi = X_b[shuffle_index[:batch_size]]
        yi = y[shuffle_index[:batch_size]]
        
        # Compute Gradient
        gradient = (2* xi.T.dot(xi.dot(theta) - yi))/(batch_size)
        
        # Compute Hessian
        H = (2*(xi**2).T.dot(xi.dot(theta))) / batch_size
        
        # Compute Hessian inverse
        Hinv = np.linalg.inv(H.T.dot(H)).dot(H.T)
        
        delta = Hinv.dot(gradient)
        
        # Update weights
        theta = theta - eta*delta
        
        
plt.plot(X, X_b.dot(theta), color='orange')
print(theta)
error = np.sum(((X_b.dot(theta) - y)**2), axis=0)/100
print('Mean squared error :', error)

