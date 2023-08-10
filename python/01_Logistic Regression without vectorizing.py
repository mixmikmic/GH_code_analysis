import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
get_ipython().magic('matplotlib inline')
print("Package Loaded")

xy = np.loadtxt("data1.txt", delimiter=',', unpack=True, dtype='float64')
train_X = xy[0:-1]
train_Y = xy[-1]

n_samples = train_X[0].size

print ""
print "Type of 'train_X' is %s" % type(train_X)
print "Shape of 'train_X' is", train_X.shape
print ("Type of 'train_Y' is ", type(train_Y))
print ("Shape of 'train_Y' is", train_Y.shape)
print ("n_samples' is", n_samples)

pos = train_Y == 1
neg = train_Y == 0

plt.figure(1)
plt.plot(train_X[0][pos], train_X[1][pos], 'r+', label='Admitted')
plt.plot(train_X[0][neg], train_X[1][neg], 'bx', label='Not admitted')
plt.axis('equal')
plt.legend(loc='lower right')

temp_X = np.insert(train_X, 0, 1, axis=0)
temp_Y = train_Y
W = np.random.random((1, 3))
print W.shape, temp_X.shape


def h(X, w):
    return np.dot(w, X)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hypothesis(X, w):
    return sigmoid(h(X, w))

def costF(X, Y, w):
    #return np.sum(-Y*np.log(hypothesis(X, w)) - (1-Y)*(np.log(1-hypothesis(X, w)))) / n_samples
    return 
    
def gradientDescent(X, Y, w, alpha, num_iters):
    for i in xrange(num_iters):
        for j in xrange(w.size):
            w[0,j] -= np.sum(np.multiply(hypothesis(temp_X, W) - temp_Y, temp_X[j,:])) * alpha / n_samples
        if i % 10000 == 0:
            print i, w, costF(X, Y, w)
    return w

finalW = gradientDescent(temp_X, temp_Y, W, 0.005, 200001)

x = np.array([np.min(temp_X[1,:]), np.max(temp_X[1,:])])
y = (-1./W[0,2])*(W[0,0] + W[0,1]*x)

plt.figure(1)
plt.plot(train_X[0][pos], train_X[1][pos], 'r+', label='Admitted')
plt.plot(train_X[0][neg], train_X[1][neg], 'bx', label='Not admitted')
plt.plot(x, y, 'r-', label='Decision Boundary')
plt.axis('equal')
plt.legend(loc='lower right')



