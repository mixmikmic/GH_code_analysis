import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('bmh')

mean = np.array([0,0])
P = np.array([[1, 1], [-1, 1]]) # kind of rotation matrix, to bend my distribution
cov = np.dot(np.dot(P, np.array([[0.1,0],[0,1]])), np.linalg.inv(P))

print cov

x, y = np.random.multivariate_normal(mean, cov, 100).T

plt.axis([-3,3,-3,3])
plt.scatter(x, y)

X = np.concatenate([x.reshape(len(x), 1), y.reshape(len(y), 1)], axis=1)

W = np.linalg.eig(np.dot(X.transpose(), X))[1]
vp = np.linalg.eig(np.dot(X.transpose(), X))[0]
print W
print vp

plt.axis("equal")
plt.axis([-3,3,-3,3])
plt.scatter(x, y)

norm = 2.5 # we want the biggest arrow to have norm 2.5
if vp[0] > vp[1]:
    coeff0 = norm
    coeff1 = norm * vp[1] / vp[0]
else:
    coeff1 = norm
    coeff0 = norm * vp[0] / vp[1]

plt.arrow(0, 0, W[0][0]*coeff0, W[1][0]*coeff0, head_width=0.2, head_length=0.2, width=0.05, fc='r', ec='r')
plt.arrow(0, 0, W[0][1]*coeff1, W[1][1]*coeff1, head_width=0.2, head_length=0.2, width=0.05, fc='k', ec='k')

Z = np.dot(X, W)
plt.axis([-3,3,-3,3])
plt.scatter(Z[:,0], Z[:,1])

print np.dot(Z.transpose(), Z)



