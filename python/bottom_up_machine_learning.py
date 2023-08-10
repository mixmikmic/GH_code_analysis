get_ipython().magic('matplotlib inline')
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt

x, y, coef = make_regression(n_samples=100, n_features=1, noise=10, coef=True)
print "True slope", coef
plt.plot(x,y,'b.')
plt.show()

def error(w, x, y):
    """ Calculate the sum of squared residuals.  The data points
        x and y that are implicit inputs to the our function e
        passed as additional inputs to the Python function error
    """
    # YOUR IMPLEMENTATION HERE
    return 0.0
    
ws = np.arange(0,100,1)
plt.plot(ws, [error(w,x,y) for w in ws])
plt.xlabel("w (slope)")
plt.ylabel("Sum of squared residuals")
plt.show

def error_grad(w, x, y):
    """ Computes the derivative of the error with respect to
        the slope w"""
    # YOUR IMPLEMENTATION HERE
    return 0.0

w_0 = 2
estimate = error_grad(w_0, x, y)
estimate

computed = 0.0     # YOUR IMPLEMENTATION HERE
estimate - computed

def gradient_descent(w, x, y, alpha, iters):
    """ Perform `iters` iterations of gradient descent
        with the initial guess w, independent variables, x,
        and dependent variables, y, and step size, alpha. """
    errors = np.zeros((iters,1))
    # WE WILL WRITE THIS TOGETHER
    return w, errors

w_f, errors = gradient_descent(w_0, x, y, .001, 100)
print w_f
plt.plot(errors)

def gradient_descent_adaptive(w, x, y, alpha, iters):
    """ Perform `iters` iterations of gradient descent
        with the initial guess w, independent variables, x,
        and dependent variables, y, and step size, alpha.
        
        The step size will automatically be adapted for good
        performance. """
    errors = np.zeros((iters,1))

    # WE WILL WRITE THIS TOGETHER
            
    return w, errors

w_f, errors = gradient_descent_adaptive(w_0, x, y, .01, 100)
print w_f
plt.plot(errors)

from load_smiles import load_smiles

def show_smiles(images, targets):
    """ Adapted from Jake Vanderplas' scikit learn tutorials. """
    fig, axes = plt.subplots(6, 6, figsize=(24, 24))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i,:].reshape((24,24)).T, cmap='gray')
        ax.text(0.05, 0.05, str(targets[i]),
                transform=ax.transAxes,
                color='green' if (targets[i]) else 'red')
        ax.set_xticks([])
        ax.set_yticks([])

data = load_smiles()
X, y = data.data, data.target
show_smiles(X, y)

def error_multi(w, X, y):
    """ Calculate the sum of squares of the residuals
        for a multivariate regression model.  w contains
        the model parameters (n_features x 1).  X contains
        a matrix of predictor variables (n_samples x n_features).
        y contains the dependent variables (n_samples x 1)"""
    # YOUR IMPLEMENTATION HERE
    return 0.0
    

w = np.zeros((576,1))
error_multi(w, data.data, data.target)

def partial_error_multi(w, X, y, j):
    """ Compute the jth partial derivative of the error function """
    # YOUR IMPLEMENTATION
    return 0.0

print "Computed partial 0", 0.0 # YOUR IMPLEMENTATION HERE
print "Analytical partial 0", partial_error_multi(w, X, y, 0)

print "Computed partial 1", 0.0 # YOUR IMPLEMENTATION HERE
print "Analytical partial 1", partial_error_multi(w, X, y, 1)

def grad_error_multi(w, X, y):
    grad = np.zeros(w.shape)
    res = X.dot(w) - y

    for i in range(X.shape[0]):
        grad += 2*res[i]*X[i,np.newaxis].T
    return grad

print grad_error_multi(w, X, y)[0:2]

def gradient_descent_multi(w, x, y, alpha, iters):
    errors = np.zeros((iters,1))
    last_error = error_multi(w, x, y)
    all_ws = []

    for i in range(iters):
        all_ws.append(w)
        grad = grad_error_multi(w, x, y)
        w_proposed = w - alpha*grad
        error_proposed = error_multi(w_proposed, x, y)
        if error_proposed < last_error:
            last_error = error_proposed
            w = w_proposed
            alpha *= 1.1
        else:
            alpha *= 0.2
        if i % 100 == 0:
            print "iter", i, "error", last_error
        errors[i] = last_error
    return w, errors, all_ws

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.2)

w_f, errors, all_ws = gradient_descent_multi(w, X_train, y_train, 10**-7, 200)
plt.plot(errors)

print np.mean((X_test.dot(w_f) > 0.5) == y_test)

from IPython.html import widgets

def image_display(i):
    plt.imshow(all_ws[i].reshape((24,24)).T, cmap='gray')

step_slider = widgets.IntSlider(min=0, max=len(all_ws)-1, value=0)
widgets.interact(image_display, i=step_slider)



