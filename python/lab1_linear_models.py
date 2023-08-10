get_ipython().magic('matplotlib inline')

import matplotlib
from matplotlib import pyplot as plt

import seaborn

import numpy as np

from numpy.linalg import inv

n = 100 # the number of samples

w1 = 1.3
w2 = 0.6
w3 = -1.5
c = 0.7

sigma = 0.3

np.random.seed(666)

def my_generate_ds(n, f, sigma=0.3):
    """
    Function used to generate synthetic data
    """
    
    X = np.random.uniform(-1, 1, size=(n,1))
    y = f(X)
    
    return X, y

f = lambda x : w1*x + w2*x**2 + w3*x**3 + c + np.random.normal(0, sigma**2, size=(n,1))


X, y = my_generate_ds(n, f)

fig, ax = plt.subplots()

ax.set_title("Dummy dataset")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.scatter(X, y, c=["orange"]);

### BEGIN STUDENTS ###

# def Phi(X, degree):
#     ...
#     ...
    
#     return X_new

# def my_fit(X, y):
#     ...
#     return w_hat

def Phi(X, degree=3):
    """
    Expand an n x 1 matrix into and n x d matrix,
    where in each column j there is X_i^j
    """
    
    n, d = X.shape
    
    l = [np.ones(shape=(n,1)), X]
    
    for j in range(2, degree+1):
        l.append(X**j)
    
    X_new = np.hstack(tuple(l))
    
    return X_new

def my_fit(X, y):
    
    w_hat = inv(X.T.dot(X)).dot(X.T).dot(y)
    
    return w_hat

### END STUDENTS ###

# Expand the original feature set
X_phi = Phi(X, 3)

# Fit the model
w_hat = my_fit(X_phi, y)

# Plot the points
fig, ax = plt.subplots()

ax.set_title("Dummy dataset")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.scatter(X, y, c=["orange"]);

# Generate points on the fitted curve
xmin = X.min()
xmax = X.max()

n_ticks = 100

ticks = np.linspace(xmin, xmax, 100).reshape((n_ticks, 1))
y_ticks = Phi(ticks).dot(w_hat)

# Plot the curve
ax.plot(ticks, y_ticks, 'k-');

n = 8 # the number of samples

w1 = 1.3
w2 = -1.5
c = -0.7

sigma = 0.3

f = lambda x : w1*x + w2*x**2 + c + np.random.normal(0, sigma**2, size=(n,1))


np.random.seed(6)
X, y = my_generate_ds(n, f)

fig, ax = plt.subplots()

ax.set_title("Misterious dataset")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.scatter(X, y, c=["orange"]);

### BEGIN STUDENTS ###

# Expand the original feature set
# Set the degree of the polynomial
# j = 1
# X_phi = ...

# Fit the model
# w_hat = ...

j = 1
X_phi = Phi(X, j)


w_hat = my_fit(X_phi, y)
### END STUDENTS ###

# Plot the points
fig, ax = plt.subplots()

ax.set_title("Fitting a polynomial of degree {}".format(j))
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.scatter(X, y, c=["orange"]);

# Generate points on the fitted curve
xmin = X.min()
xmax = X.max()

n_ticks = 100

ticks = np.linspace(xmin, xmax, 100).reshape((n_ticks, 1))
y_ticks = Phi(ticks, j).dot(w_hat)

# Plot the curve
ax.plot(ticks, y_ticks, 'k-');

### BEGIN STUDENTS ###

# Expand the original feature set
# Set the degree of the polynomial
# j = ...
# X_phi = ...

# Fit the model
# w_hat = ...

j = 2
X_phi = Phi(X, j)

w_hat = my_fit(X_phi, y)

### END STUDENTS ###



# Plot the points
fig, ax = plt.subplots()

ax.set_title("Fitting a polynomial of degree {}".format(j))
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.scatter(X, y, c=["orange"]);

# Generate points on the fitted curve
xmin = X.min()
xmax = X.max()

n_ticks = 100

ticks = np.linspace(xmin, xmax, 100).reshape((n_ticks, 1))
y_ticks = Phi(ticks, j).dot(w_hat)

# Plot the curve
ax.plot(ticks, y_ticks, 'k-');

### BEGIN STUDENTS ###

# Expand the original feature set
# Set the degree of the polynomial
# j = ...
# X_phi = ...

# Fit the model
# w_hat = ...

j = 3
X_phi = Phi(X, j)

w_hat = my_fit(X_phi, y)

### END STUDENTS ###


# Plot the points
fig, ax = plt.subplots()

ax.set_title("Fitting a polynomial of degree {}".format(j))
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.scatter(X, y, c=["orange"]);

# Generate points on the fitted curve
xmin = X.min()
xmax = X.max()

n_ticks = 100

ticks = np.linspace(xmin, xmax, 100).reshape((n_ticks, 1))
y_ticks = Phi(ticks, j).dot(w_hat)

# Plot the curve
ax.plot(ticks, y_ticks, 'k-');

print(w_hat)

### BEGIN STUDENTS ###

# Expand the original feature set
# Set the degree of the polynomial
# j = ...
# X_phi = ...

# Fit the model
# w_hat = ...

j = 7
X_phi = Phi(X, j)

w_hat = my_fit(X_phi, y)

### END STUDENTS ###

# Plot the points
fig, ax = plt.subplots()

ax.set_title("Fitting a polynomial of degree {}".format(j))
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.scatter(X, y, c=["orange"]);

# Generate points on the fitted curve
xmin = X.min()
xmax = X.max()

n_ticks = 100

ticks = np.linspace(xmin, xmax, 100).reshape((n_ticks, 1))
y_ticks = Phi(ticks, j).dot(w_hat)

# Plot the curve
ax.plot(ticks, y_ticks, 'k-');



