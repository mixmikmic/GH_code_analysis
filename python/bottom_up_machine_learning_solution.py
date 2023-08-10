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
    return np.sum([(y[i] - w*x[i,0])**2 for i in range(len(x))])

ws = np.arange(0,100,1)
plt.plot(ws, [error(w,x,y) for w in ws])
plt.xlabel("w (slope)")
plt.ylabel("Sum of squared residuals")
plt.show()

def error_grad(w, x, y):
    """ Computes the derivative of the error with respect to
        the slope w"""
    return np.sum([2*(y[i] - w*x[i,0])*(-x[i,0]) for i in range(len(x))])

w_0 = 2
estimate = error_grad(w_0, x, y)
estimate

computed = (error(w_0+10**-6, x, y) - error(w_0, x, y))/10**-6
estimate - computed

def gradient_descent(w, x, y, alpha, iters):
    errors = np.zeros((iters,1))
    for i in range(iters):
        w = w - alpha*error_grad(w, x, y)
        errors[i] = error(w, x, y)
    return w, errors

w_f, errors = gradient_descent(w_0, x, y, .001, 100)
print w_f
plt.plot(errors)

def gradient_descent(w, x, y, alpha, iters):
    errors = np.zeros((iters,1))
    last_error = error(w, x, y)
    for i in range(iters):
        w_proposed = w - alpha*error_grad(w, x, y)
        error_proposed = error(w_proposed, x, y)
        if error_proposed < last_error:
            last_error = error_proposed
            w = w_proposed
            alpha *= 1.5
        else:
            alpha *= 0.6
        errors[i] = last_error
            
    return w, errors

w_f, errors = gradient_descent(w_0, x, y, .01, 100)
print w_f
plt.plot(errors)

from load_smiles import load_smiles

def show_smiles(images, targets):
    """ Adapted from Jake Vanderplas' scikit learn tutorials. """
    fig, axes = plt.subplots(6, 6, figsize=(24, 24))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i,:-1].reshape((24,24)).T, cmap='gray')
        ax.text(0.05, 0.05, str(targets[i]),
                transform=ax.transAxes,
                color='green' if (targets[i]) else 'red')
        ax.set_xticks([])
        ax.set_yticks([])

data = load_smiles()
X, y = data.data, data.target
X = np.hstack((X, np.ones((X.shape[0],1))))
show_smiles(X, y)

def error_multi(w, X, y):
    predictions = X.dot(w)
    return np.sum((predictions - y)*(predictions - y))

w = np.zeros((X.shape[1],1))
get_ipython().magic('timeit error_multi(w, X, y)')

def partial_error_multi(w, X, y, j):
    """ Compute the partial error_multi with respect to w_j """
    partial = 0
    for i in range(X.shape[0]):
        prediction = X[i,:].T.dot(w)
        partial += 2*(prediction - y[i])*X[i,j]
    return partial

e_0 = np.zeros(w.shape)
e_0[0] = 10**-8

print "Computed partial 0", (error_multi(w+e_0, X, y) - error_multi(w, X, y))/e_0[0]
print "Analytical partial 0", partial_error_multi(w, X, y, 0)

e_1 = np.zeros(w.shape)
e_1[1] = 10**-8

print "Computed partial 1", (error_multi(w+e_1, X, y) - error_multi(w, X, y))/e_1[1]
print "Analytical partial 1", partial_error_multi(w, X, y, 1)

def grad_error_multi_faster(w, X, y):
    """ Compute the gradient of error_multi with respect to w """
    grad = np.zeros(w.shape)
    res = X.dot(w) - y
    for i in range(X.shape[0]):
        grad += 2*res[i]*X[i,np.newaxis].T
    return grad

get_ipython().magic('timeit grad_error_multi_faster(w, X, y)')

def grad_error_multi(w, X, y):
    """ Compute the gradient of error_multi with respect to w
        in a more efficient manner. """
    grad = np.zeros(w.shape)
    res = X.dot(w) - y
    grad = 2*res.T.dot(X).T
    return grad
e_0 = np.zeros(w.shape)
e_0[0] = 10**-8

print "Computed partial 0", (error_multi(w+e_0, X, y) - error_multi(w, X, y))/e_0[0]
print "Analytical partial 0", grad_error_multi(w, X, y)[0]

e_1 = np.zeros(w.shape)
e_1[1] = 10**-8

print "Computed partial 1", (error_multi(w+e_1, X, y) - error_multi(w, X, y))/e_1[1]
print "Analytical partial 1", grad_error_multi(w, X, y)[1]
get_ipython().magic('timeit grad_error_multi(w, X, y)')

def gradient_descent_multi(w, x, y, alpha, iters):
    """ Perform `iters` iterations of gradient descent
        alpha is the step size to start with (this will be adapted)
        w is a dx1 numpy array containing an initial guess for the parameters
        x is a nxd numpy array containing the independent variables
        y is a nx1 numpy array containing the dependent variable 
        
        returns: the fitted value of w, the error for each iteration,
                 and a list containing the value of w at each iteration. """
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
            alpha *= 0.8
        if i % 100 == 0:
            print "iter", i, "error", last_error
        errors[i] = last_error
    return w, errors, all_ws

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5)
w = np.zeros(w.shape)
w_f, errors, all_ws = gradient_descent_multi(w,
                                             X_train,
                                             y_train,
                                             10**-7,
                                             1000)
plt.plot(errors)
plt.xlabel('iteration')
plt.ylabel('error')

def rprop_descent_multi(w, x, y, alpha, iters):
    """ Perform `iters` iterations of rprop https://en.wikipedia.org/wiki/Rprop
        alpha is the step size to start with (this will be adapted per feature)
        w is a dx1 numpy array containing an initial guess for the parameters
        x is a nxd numpy array containing the independent variables
        y is a nx1 numpy array containing the dependent variable 
        
        returns: the fitted value of w, the error for each iteration,
                 and a list containing the value of w at each iteration. """
    g = alpha*np.ones(w.shape)
    errors = np.zeros((iters,1))
    last_error = error_multi(w, x, y)
    all_ws = []

    for i in range(iters):
        all_ws.append(np.copy(w))
        grad = grad_error_multi(w, x, y)
        w_proposed = w - g*grad
        grad_proposed = grad_error_multi(w_proposed, x, y)
        mask = np.sign(grad) == np.sign(grad_proposed)
        w[mask] = w_proposed[mask]
        g[mask] *= 1.1
        g[~mask] *= 0.8

        errors[i] = error_multi(w, x, y)
        if i % 100 == 0:
            print "iter", i, "error", errors[i]
    return w, errors, all_ws

w = np.zeros(w.shape)
w_f_rprop, errors_rprop, all_ws = rprop_descent_multi(w,
                                                      X_train,
                                                      y_train,
                                                      10**-6.5,
                                                      1000)
plt.plot(errors)
plt.plot(errors_rprop)
plt.legend(['Vanilla GD', 'rprop'])
plt.xlabel('iteration')
plt.ylabel('error')

def make_sequences(T):
    """ Create the sequences needed for Nesterov's accelerated
        gradient descent. """
    lambdas = np.zeros((T,))
    gammas = np.zeros((T-1,))
    lambdas[0] = 0.0
    for t in range(1,T):
        lambdas[t] = (1. + (1 + 4*lambdas[t-1]**2)**0.5)/2.0
        gammas[t-1] = (1 - lambdas[t-1])/lambdas[t]
    return lambdas, gammas

def nesterov_descent_multi(w, x, y, beta, iters):
    """ Perform `iters` iterations of Nesterov accelerated gradient descent
        https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/
        The parameter beta can be thought of
        as 1 over the step size.
        w is a dx1 numpy array containing an initial guess for the parameters
        x is a nxd numpy array containing the independent variables
        y is a nx1 numpy array containing the dependent variable 
        
        returns: the fitted value of w, the error for each iteration,
                 and a list containing the value of w at each iteration. """
    lambdas, gammas = make_sequences(iters+1)
    errors = np.zeros((iters,1))
    all_ws = []
    all_ys = [np.copy(w)]

    for i in range(iters):
        all_ws.append(np.copy(w))
        grad = grad_error_multi(w, x, y)
        y_s = w - 1./beta*grad
        w = (1 - gammas[i])*y_s + gammas[i]*all_ys[-1]
        all_ys.append(np.copy(y_s))

        errors[i] = error_multi(w, x, y)
        if i % 100 == 0:
            print "iter", i, "error", errors[i]
    return w, errors, all_ws

w = np.zeros(w.shape)
w_f_nesterov, errors_nesterov, all_ws = nesterov_descent_multi(w,
                                                               X_train,
                                                               y_train,
                                                               10**6.45,
                                                               1000)
plt.plot(errors)
plt.plot(errors_rprop)
plt.plot(errors_nesterov)
plt.legend(['Vanilla GD', 'rprop', 'Nesterov Accelerated'])

def conjugate_descent_multi(w, x, y, iters):
    """ Perform `iters` iterations of conjugate gradient descent
        https://en.wikipedia.org/wiki/Conjugate_gradient_method
        w is a dx1 numpy array containing an initial guess for the parameters
        x is a nxd numpy array containing the independent variables
        y is a nx1 numpy array containing the dependent variable 
        
        returns: the fitted value of w, the error for each iteration,
                 and a list containing the value of w at each iteration. """
    errors = np.zeros((iters,1))
    A = x.T.dot(x)
    b = x.T.dot(y)
    r = b - A.dot(w)
    p = np.copy(r)
    k = 0
    for i in range(iters):
        alpha_k = r.T.dot(r) / (p.T.dot(A).dot(p))
        w = w + alpha_k*p
        r_old = np.copy(r)
        r = r - alpha_k*A.dot(p)
        beta_k = (r.T.dot(r))/(r_old.T.dot(r_old))
        p = r + beta_k*p
        k = k + 1
        errors[i] = error_multi(w, x, y)
        if i % 100 == 0:
            print "iter", i, "error", errors[i]
    return w, errors, all_ws

w = np.zeros(w.shape)
w_f_conjugate, errors_conjugate, all_ws = conjugate_descent_multi(w,
                                                                  X_train,
                                                                  y_train,
                                                                  1000)
plt.plot(errors)
plt.plot(errors_rprop)
plt.plot(errors_nesterov)
plt.plot(errors_conjugate)
plt.legend(['Vanilla GD', 'rprop', 'Nesterov Accelerated', 'Conjugate Gradient'])

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print "Optimal error", np.sum((model.predict(X_train) - y_train)**2)

print "adaptive", np.mean((X_test.dot(w_f) > 0.5) == y_test)
print "rmsprop", np.mean((X_test.dot(w_f_rprop) > 0.5) == y_test)
print "nesterov", np.mean((X_test.dot(w_f_nesterov) > 0.5) == y_test)
print "conjugate", np.mean((X_test.dot(w_f_conjugate) > 0.5) == y_test)

from IPython.html import widgets

def image_display(i):
    plt.imshow(all_ws[i][:-1].reshape((24,24)).T, cmap='gray')

step_slider = widgets.IntSlider(min=0, max=len(all_ws)-1, value=0)
widgets.interact(image_display, i=step_slider)



