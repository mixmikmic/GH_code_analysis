get_ipython().magic('matplotlib inline')

import matplotlib
from matplotlib import pyplot as plt

import seaborn

import numpy as np

from numpy.linalg import inv

from sklearn.metrics import mean_squared_error

def my_generate_ds(n, f, sigma=0.3):
    """
    Function used to generate synthetic data
    """
    
    X = np.random.uniform(-1, 1, size=(n,1))
    y = f(X)
    
    return X, y

# Define the feature mapping
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
    """
    Fit a linear model
    """
    
    w_hat = inv(X.T.dot(X)).dot(X.T).dot(y)
    
    return w_hat

### Dataset generation

n = 8 # the number of samples

w1 = 1.3
w2 = -1.5
c = -0.7

sigma = 0.3

np.random.seed(6)

f = lambda x : w1*x + w2*x**2 + c + np.random.normal(0, sigma**2, size=(n,1))

X, y = my_generate_ds(n, f)

#####################
### Model fitting ###
#####################


### BEGIN STUDENTS ###

# ### Straight line
# X_phi1 = ...
# w_hat1 = ...

# ### Polynomial of degree 2
# X_phi2 = ...
# w_hat2 = ...

# ### Polynomial of degree 3
# X_phi3 = ...
# w_hat3 = ...

# ### Polynomial of degree 7
# X_phi7 = ...
# w_hat7 = ...

### Straight line
X_phi1 = Phi(X, 1)
w_hat1 = my_fit(X_phi1, y)

### Polynomial of degree 2
X_phi2 = Phi(X, 2)
w_hat2 = my_fit(X_phi2, y)

### Polynomial of degree 3
X_phi3 = Phi(X, 3)
w_hat3 = my_fit(X_phi3, y)

### Polynomial of degree 7
X_phi7 = Phi(X, 7)
w_hat7 = my_fit(X_phi7, y)

### END STUDENTS ###

######################
### Curve plotting ###
######################

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

y_ticks1 = Phi(ticks, 1).dot(w_hat1)
y_ticks2 = Phi(ticks, 2).dot(w_hat2)
y_ticks3 = Phi(ticks, 3).dot(w_hat3)
y_ticks7 = Phi(ticks, 7).dot(w_hat7)

# Plot the fitted curves
curve1 = ax.plot(ticks, y_ticks1, 'r-', label="Straight line", alpha=0.5);
curve2 = ax.plot(ticks, y_ticks2, 'g-', label="Polynomial of degree 2", alpha=0.5);
curve3 = ax.plot(ticks, y_ticks3, 'b-', label="Polynomial of degree 3", alpha=0.5);
curve7 = ax.plot(ticks, y_ticks7, 'y-', label="Polynomial of degree 7", alpha=0.5);

ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);

### Using the mean_squared_error function, compute the MSE for all four models

### REMINDER ###
# You already have the models for all curves, saved in the four variables w_hat<M>, with <M> = {1,2,3,7}
# 

### BEGIN STUDENTS ###
# yhat1 = ...
# yhat2 = ...
# yhat3 = ...
# yhat7 = ...

# mse1 = ...
# mse2 = ...
# mse3 = ...
# mse7 = ...

yhat1 = X_phi1.dot(w_hat1)
yhat2 = X_phi2.dot(w_hat2)
yhat3 = X_phi3.dot(w_hat3)
yhat7 = X_phi7.dot(w_hat7)

mse1 = mean_squared_error(y, yhat1)
mse2 = mean_squared_error(y, yhat2)
mse3 = mean_squared_error(y, yhat3)
mse7 = mean_squared_error(y, yhat7)
### END STUDENTS ###

print("MSE for straight line model\t: {:1.5f}".format(mse1))
print("MSE for polynomial of degree 2\t: {:1.5f}".format(mse2))
print("MSE for polynomial of degree 3\t: {:1.5f}".format(mse3))
print("MSE for polynomial of degree 7\t: {:1.3e}".format(mse7))

n = 20

f = lambda x : w1*x + w2*x**2 + c + np.random.normal(0, sigma**2, size=(n,1))

np.random.seed(9)

X_new, y_new = my_generate_ds(n, f)

######################
### Curve plotting ###
######################

fig, ax = plt.subplots()

ax.set_title("Dummy dataset")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

# Plot new points
ax.scatter(X_new, y_new, c=["green"]);


# Generate points on the fitted curve
xmin = X_new.min()
xmax = X_new.max()

n_ticks = 100

ticks = np.linspace(xmin, xmax, 100).reshape((n_ticks, 1))

y_ticks1 = Phi(ticks, 1).dot(w_hat1)
y_ticks2 = Phi(ticks, 2).dot(w_hat2)
y_ticks3 = Phi(ticks, 3).dot(w_hat3)
y_ticks7 = Phi(ticks, 7).dot(w_hat7)

# Plot the fitted curves
curve1 = ax.plot(ticks, y_ticks1, 'r-', label="Straight line", alpha=0.5);
curve2 = ax.plot(ticks, y_ticks2, 'g-', label="Polynomial of degree 2", alpha=0.5);
curve3 = ax.plot(ticks, y_ticks3, 'b-', label="Polynomial of degree 3", alpha=0.5);
curve7 = ax.plot(ticks, y_ticks7, 'y-', label="Polynomial of degree 7", alpha=0.5);

ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);

### Using the mean_squared_error function, compute the MSE for all four models

### REMINDER ###
# You already have the models for all curves, saved in the four variables w_hat<M>, with <M> = {1,2,3,7}
# To predict the outputs relative to the input you have to compute the dot product between 
# the appropriate feature mapping for the model used (using function Phi) and the model itself.

### BEGIN STUDENTS ###

# yhat1 = ...
# yhat2 = ...
# yhat3 = ...
# yhat7 = ...

# mse1 = ...
# mse2 = ...
# mse3 = ...
# mse7 = ...

yhat1 = Phi(X_new, 1).dot(w_hat1)
yhat2 = Phi(X_new, 2).dot(w_hat2)
yhat3 = Phi(X_new, 3).dot(w_hat3)
yhat7 = Phi(X_new, 7).dot(w_hat7)

mse1 = mean_squared_error(y_new, yhat1)
mse2 = mean_squared_error(y_new, yhat2)
mse3 = mean_squared_error(y_new, yhat3)
mse7 = mean_squared_error(y_new, yhat7)
### END STUDENTS ###

print("MSE for straight line model (new data)\t\t: {:1.5f}".format(mse1))
print("MSE for polynomial of degree 2 (new data)\t: {:1.5f}".format(mse2))
print("MSE for polynomial of degree 3 (new data)\t: {:1.5f}".format(mse3))
print("MSE for polynomial of degree 7 (new data)\t: {:1.4f}".format(mse7))

### Here is a dataset of 24 samples. Split it in two (I suggest 2/3 for training and 1/3 for test), 
### train the usual four models on the training test, plot the curves (copy the code from above) and
### compute the mean squared error for all four models.

n = 24

f = lambda x : w1*x + w2*x**2 + c + np.random.normal(0, sigma**2, size=(n,1))

X, y = my_generate_ds(n, f)

### BEGIN STUDENTS ###
# X_train = X[:16, :]
# X_test = X[16:, :]

# y_train = X[:16, :]
# y_test = X[16:, :]
### END STUDENTS ###



