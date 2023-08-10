import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

mean = np.array([0,0])
P = np.array([[1, 1], [-1, 1]]) # matrice de changement de base, pour pencher ma distribution
cov = np.dot(np.dot(P, np.array([[0.1,0],[0,1]])), np.linalg.inv(P))
print cov

x, y = np.random.multivariate_normal(mean, cov, 100).T

plt.axis([-3,3,-3,3])
plt.scatter(x, y)

one = np.ones(len(x)).reshape(len(x), 1) # for the bias

X, y = np.concatenate([one, x.reshape(len(x), 1)], axis=1) , y.reshape(len(y), 1)

S = np.dot(X.transpose(), X)
lamda = 0.9
print S

w = np.dot(np.dot(np.linalg.inv(S + lamda * np.eye(len(S))), X.transpose()), y)
print w

new_x = np.linspace(-5, 5, 101)
y_predict = w[1] * new_x + w[0]

plt.axis([-3,3,-3,3])
plt.scatter(x, y)
plt.plot(new_x, y_predict)



