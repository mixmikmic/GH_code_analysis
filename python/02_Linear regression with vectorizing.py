import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
print("Package Loaded")

np.random.seed(1)
def f(x, a, b):
    n = train_X.size
    vals = np.zeros((1, n))
    for i in range(0, n):
        ax = np.multiply(a, x.item(i))
        val = np.add(ax, b)
        vals[0, i] = val
    return vals

Wref = 0.7
bref = -1.
n = 20
noise_var = 0.001
train_X = np.random.random((1, n))
ref_Y = f(train_X, Wref, bref)
train_Y = ref_Y + np.sqrt(noise_var)*np.random.randn(1, n)
n_samples = train_X.size

print ""
print "Type of 'train_X' is %s" % type(train_X)
print "Shape of 'train_X' is", train_X.shape
print ("Type of 'train_Y' is ", type(train_Y))
print ("Shape of 'train_Y' is", train_Y.shape)

plt.figure(1)
plt.plot(train_X[0, :], ref_Y[0, :], 'ro', label='Original data')
plt.plot(train_X[0, :], train_Y[0, :], 'bo', label='Training data')
plt.axis('equal')
plt.legend(loc='lower right')

temp_X = np.insert(train_X, 0, 1, axis=0).T
temp_Y = train_Y.T
W = np.random.random((2, 1))

def h(X, w):
    return np.dot(X, w)

def costF(X, w, Y):
    #sqrErrors = np.power(h(X, w) - Y, 2)
    #return np.sum(sqrErrors) / (2*n_samples)    
    return float(np.dot((h(X, w) - Y).T, (h(X, w) - Y)) / (2*n_samples))

def gradientDescent(X, Y, w, alpha, num_iters):
    for i in xrange(num_iters):
        #w[0] -= np.sum(np.multiply(h(X, w) - Y, X[:,0].reshape(n_samples, 1))) * alpha / n_samples
        #w[1] -= np.sum(np.multiply(h(X, w) - Y, X[:,1].reshape(n_samples, 1))) * alpha / n_samples
        for j in xrange(w.size):
            w[j] -= np.sum(np.multiply(h(X, w) - Y, np.array(X[:,j].reshape(n_samples, 1)))) * alpha / n_samples
        if i % 1000 == 0:
            print i, w.T, costF(X, w, Y)
    return w

finalW = gradientDescent(temp_X, temp_Y, W, 0.01, 20000)
print finalW.T

plt.figure(1)
plt.plot(train_X[0, :], ref_Y[0, :], 'ro', label='Original data')
plt.plot(train_X[0, :], train_Y[0, :], 'bo', label='Training data')
plt.plot(train_X[0, :], h(temp_X, W).T[0, :], 'k', label='Fitting Line')
plt.axis('equal')
plt.legend(loc='lower right')



