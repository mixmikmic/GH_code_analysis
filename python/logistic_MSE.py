# Import all required libraries
from __future__ import division # For python 2.*

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

np.random.seed(0)
get_ipython().run_line_magic('matplotlib', 'inline')

lc2_data = np.genfromtxt('data/lc2_data.txt', delimiter=None)
X, Y = lc2_data[:, :-1], lc2_data[:, -1]

Y[np.where(Y == -1)[0]] = 0
ml.plotClassify2D(None, X, Y)

def add_const(X):
    """Adding constant intercept to data. """
    return np.hstack([np.ones([X.shape[0], 1]), X])

def sigmoid(z):
    """Sigmoid function. """
    return 1. / (1 + np.exp(-z))

def response(x, theta):
    """Inner product of theta x"""
    return np.dot(x, theta)

def sig_pred(x, theta):
    """Returns the class according to the threshold 0.5"""
    f_vals = sigmoid(response(x, theta))
    f_vals[f_vals >= .5] = 1
    f_vals[f_vals < .5] = 0

    return f_vals

def mse_err(x, y, theta):
    """Calculates MSE for Logistic Classifier. """
    return np.mean((sigmoid(response(x, theta)) - y)**2)

Xconst = add_const(X)
theta = [5., 1., 1.]
print mse_err(Xconst, Y, theta)

error = sigmoid(response(Xconst, theta)) - Y

sig_resp = sigmoid(response(Xconst, theta))
der_sigmoid = sig_resp * (1 - sig_resp)

tmp = error * der_sigmoid

grad_vals = np.reshape(tmp, [-1, 1]) * Xconst 

grad_vals.shape

gradient = np.mean(grad_vals, axis=0)
# Notice that I ignored the 2. It's totally ok to do that.

gradient

def mse_grad(Xconst, Y, theta):
    """Calculates the Logistic MSE Gradient. """
    f_val = sigmoid(response(Xconst, theta))
    error = f_val - Y
    der_sigmoid = f_val * (1 - f_val)
    
    grad_vals =  np.reshape(error * der_sigmoid, [-1, 1]) * Xconst 
    
    return np.mean(grad_vals, axis=0)

print mse_grad(Xconst, Y, theta)

def train(X, Y, theta, a=0.05, tol=1e-3, max_iters=100):
    """Trains the model and returns an array with the error at each iteration. """
    # In this version we assume we get the regular X as an input
    Xconst = add_const(X)
    
    J_err = [np.inf]
    for i in xrange(max_iters):
        # Updating the theta
        theta -= a * mse_grad(Xconst, Y, theta)
        
        # Computing the new error
        error = mse_err(Xconst, Y, theta)
        J_err.append(error)
        
        # Checking for stopping conditions. Notice that this time I'm forcing at least
        # 10 iterations.
        if i > 10 and np.abs(J_err[-2] - J_err[-1]) < tol:
            break
        
    return theta, J_err 

theta = np.array([-5., 1., 0.])
_, err = train(X, Y, theta)

plt.plot(err)

from mltools.base import classifier
class LogisticMSE(classifier):
    def __init__(self, theta=None):
        self.theta = theta

    def add_const(self, X):
        return np.hstack([np.ones([X.shape[0], 1]), X])        
        
    # Main functions
    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))    

    def response(self, x):
        return np.dot(x, self.theta)
    
    def sig_pred(self, x):
        f_vals = self.sigmoid(self.response(x))
        f_vals[f_vals >= .5] = 1
        f_vals[f_vals < .5] = 0

        return f_vals    
    
    def predict(self, X):
        Xconst = np.atleast_2d(X)
        if Xconst.shape[1] == self.theta.shape[0] - 1:
            Xconst = self.add_const(Xconst)
        
        return self.sig_pred(Xconst)

    def mse_err(self, x, y):
        """Calculate mse error for logistic regression. """
        return np.mean((self.sigmoid(self.response(x)) - y)**2)    
    
    def mse_grad(self, Xconst, Y):
        """Calculate gradient of loss/cost for logistic MSE. """
        f_val = self.sigmoid(self.response(Xconst))
        error = f_val - Y
        der_sigmoid = f_val * ( 1 - f_val )

        tmp = error * der_sigmoid
        derivative_sum = np.reshape(tmp, [-1, 1]) * Xconst

        return np.mean(derivative_sum, axis=0)    
    
    def train(self, X, Y, a=0.01, tol=1e-3, max_iters=100):
        """Trains the model and returns an array with the error at each iteration. """
        # In this version we assume we get the regular X as an input
        Xconst = self.add_const(X)
        if self.theta is None:
            print 'Initializing theta'
            self.theta = Xconst[5]
#             self.theta = np.random.rand(Xconst.shape[1])        
        
        J_err = [np.inf]
        for i in xrange(max_iters):
            # Updating the theta
            self.theta -= a * self.mse_grad(Xconst, Y)
            
            # Computing the new error
            error = self.mse_err(Xconst, Y)
            J_err.append(error)
            
            # Checking for stopping conditions. Notice that this time I'm forcing at least
            # 10 iterations.
            if i > 10 and np.abs(J_err[-2] - J_err[-1]) < tol:
                break
        
        print("Error at end of iteration {0} is {1}".format(i, J_err[-1]))   
        return J_err

lc = LogisticMSE(theta=np.array([-5., 1., 0.]))
ml.plotClassify2D(lc, X, Y)

J_err = lc.train(X, Y, a=0.1, max_iters=100)
ml.plotClassify2D(lc, X, Y)

from IPython import display
for i in range(2, len(J_err)):
    plt.figure(1)
    plt.plot(J_err[:i], 'k-', lw=3)
    plt.draw()
    
    # Waiting on either an input or time sleep.
    _ = raw_input()
    
    # Just add this to the original code where you think it should go :)
    display.clear_output()
    display.display(plt.gcf())    

