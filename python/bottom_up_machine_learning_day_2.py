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
import numpy as np

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

from IPython.html import widgets

def image_display(i):
    plt.imshow(all_ws[i].reshape((24,24)).T, cmap='gray')

step_slider = widgets.IntSlider(min=0, max=len(all_ws)-1, value=0)
widgets.interact(image_display, i=step_slider)



