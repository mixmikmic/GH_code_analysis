import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# Define the function f and its derivative
def f(x):
    return 0.5*(x[0]**2+10*x[1]**2)

def df(x):
    return np.array([x[0], 10*x[1]])

def ddf(x):
    return np.array([[1, 0], [0, 10]])

# Get a list of values for specifying the level sets
xvals = np.array([[np.linspace(-15,-2,20)], [np.zeros(20)]])
yvals = list(reversed(f(xvals)[0]))

# Create a meshgrid and a contour plot
xx = np.linspace(-10,10,100)
yy = np.linspace(-4,4,100)
X, Y = np.meshgrid(xx, yy)
# The construction inside looks odd: we want to transform the set of input pairs given
# by the meshgrid into a 2 x n array of values that we can apply f to (calling f on such
# an array will apply the function f to each column)
Z = f(np.dstack((X,Y)).reshape((X.size, 2)).transpose())
# the result of applying f is a long list, but we want a matrix
Z = Z.reshape(X.shape)

get_ipython().magic('matplotlib inline')
cmap = plt.cm.get_cmap("coolwarm")

# Now apply gradient descent and newton's method
import numpy.linalg as la

# The implementations below return a whole trajectory, instead of just the final result

def graddesc(f, df, ddf, x0, tol, maxiter=100):
    """
    Gradient descent for quadratic function
    """
    x = np.vstack((x0+2*tol*np.ones(x0.shape),x0)).transpose()
    i = 1
    while ( la.norm(x[:,i]-x[:,i-1]) > tol ) and ( i < maxiter ):
        r = df(x[:,i])
        alpha = np.dot(np.dot(ddf(x[:,i]),x[:,i]),r)/np.dot(r,np.dot(ddf(x[:,i]),r))
        xnew = x[:,i] - alpha*r
        x = np.concatenate((x,xnew.reshape((2,1))), axis=1)
        i += 1
    return x[:,1:]

def newton(f, df, ddf, x0, tol, maxiter=100):
    """
    Newton's method
    """
    x = np.vstack((x0+2*tol*np.ones(x0.shape),x0)).transpose()
    i = 1
    while ( la.norm(x[:,i]-x[:,i-1]) > tol ) and ( i < maxiter ):
        grad = df(x[:,i])
        hess = ddf(x[:,i])
        z = la.solve(hess,grad)
        xnew = x[:,i]-z
        x = np.concatenate((x,xnew.reshape((2,1))), axis=1)
        i += 1
    return x[:,1:]

x0 = np.array([10.,1.])
tol = 1e-8

xg = graddesc(f, df, ddf, x0, tol)
xn = newton(f, df, ddf, x0, tol)

plt.subplot(1,2,1)
plt.contour(X, Y, Z, yvals, cmap = cmap)
plt.plot(xg[0,:], xg[1,:], color='black', linewidth=3)

plt.subplot(1,2,2)
plt.contour(X, Y, Z, yvals, cmap = cmap)
plt.plot(xn[0,:], xn[1,:], color='black', linewidth=3)

plt.show()



