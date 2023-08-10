get_ipython().magic('matplotlib inline')

import matplotlib
from matplotlib import pyplot as plt

import seaborn

import numpy as np

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
    Expand an n x 1 matrix into and n x (d-1) matrix,
    where in each column j there is X_i^(j+1)
    """
    
    n, d = X.shape
    
    l = [ X]
    
    for j in range(2, degree+1):
        l.append(X**j)
    
    X_new = np.hstack(tuple(l))
    
    return X_new

### Dataset generation

n = 8 # the number of samples

w1 = 1.3
w2 = -1.5
c = -0.7

sigma = 0.3

np.random.seed(6)

f = lambda x : w1*x + w2*x**2 + c + np.random.normal(0, sigma**2, size=(n,1))

X, y = my_generate_ds(n, f)

from sklearn.linear_model import LinearRegression

### BEGIN STUDENTS ###

# w_hat1 = ...
# c1 = ...
# ...
# w_hat7 = ...
# c7 = ...

reg1 = LinearRegression()
reg1.fit(X,y)
w_hat1 = reg1.coef_.ravel()
c1 = reg1.intercept_

reg2 = LinearRegression()
reg2.fit(Phi(X,2),y)
w_hat2 = reg2.coef_.ravel()
c2 = reg2.intercept_

reg3 = LinearRegression()
reg3.fit(Phi(X,3),y)
w_hat3 = reg3.coef_.ravel()
c3 = reg3.intercept_

reg7 = LinearRegression()
reg7.fit(Phi(X,7),y)
w_hat7 = reg7.coef_.ravel()
c7 = reg7.intercept_

### END STUDENTS ###

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

# Something has changed here...
y_ticks1 = ticks.dot(w_hat1) + c1
y_ticks2 = Phi(ticks, 2).dot(w_hat2) + c2
y_ticks3 = Phi(ticks, 3).dot(w_hat3) + c3
y_ticks7 = Phi(ticks, 7).dot(w_hat7) + c7

# Plot the fitted curves
curve1 = ax.plot(ticks, y_ticks1, 'r-', label="Straight line", alpha=0.5);
curve2 = ax.plot(ticks, y_ticks2, 'g-', label="Polynomial of degree 2", alpha=0.5);
curve3 = ax.plot(ticks, y_ticks3, 'b-', label="Polynomial of degree 3", alpha=0.5);
curve7 = ax.plot(ticks, y_ticks7, 'y-', label="Polynomial of degree 7", alpha=0.5);

ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);


### In case you want to be extra-sure, grab the code used to compute the w_hat for the 
### polynomial of degree 2 in the previous lab and compare the results using the two procedures
### However, if you do so, you will need to include again the column of 'ones' manually 
### (check the documentation for np.hstack)

from sklearn.linear_model import Ridge

### BEGIN STUDENTS ###

# j = ...
# my_alpha = ...
# reg = ... 

j = 7
my_alpha = 1e-1

reg = Ridge(alpha=my_alpha)
reg.fit(Phi(X,j),y)

### END STUDENTS ###

### PLOTTING
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

# Something has changed here...
y_ticks = reg.predict(Phi(ticks, j))


# Plot the fitted curves
curve1 = ax.plot(ticks, y_ticks, 'r-', label="Straight line", alpha=0.5);

### BEGIN STUDENTS ###

# j = ...
# alpha_low = ...
# alpha_high = ...

# reg_low = ...
# reg_high = ...

j = 7

alpha_low = 1e-20

reg_low = Ridge(alpha=alpha_low)
reg_low.fit(Phi(X,j),y)

alpha_high = 1e5

reg_high = Ridge(alpha=alpha_high)
reg_high.fit(Phi(X,j),y)

### END STUDENTS ###

### PLOTTING

# Generate points on the fitted curve
xmin = X.min()
xmax = X.max()

n_ticks = 100
ticks = np.linspace(xmin, xmax, 100).reshape((n_ticks, 1))

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,5), sharey=True)

ax1.set_title(r"Low $\alpha$")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")

ax1.scatter(X, y, c=["orange"]);


ax2.set_title(r"High $\alpha$")
ax2.set_xlabel("$x$")
ax2.set_ylabel("$y$")

ax2.scatter(X, y, c=["orange"]);

# Compute the y coordinates of the points of the curves
y_ticks_low = reg_low.predict(Phi(ticks, j))
y_ticks_high = reg_high.predict(Phi(ticks, j))

# Plot the fitted curves
curve1 = ax1.plot(ticks, y_ticks_low, 'r-', label="Straight line", alpha=0.5);
curve2 = ax2.plot(ticks, y_ticks_high, 'r-', label="Straight line", alpha=0.5);

print("Intercept of the fitted curve:\t{}".format(reg_high.intercept_))
print("Mean of the output values y:\t{}".format(y.mean()))

from sklearn.metrics import mean_squared_error

### Dataset generation

n = 15 # the number of samples

w1 = 1.3
w2 = -1.5
c = -0.7

sigma = 0.3

np.random.seed(6)

f = lambda x : w1*x + w2*x**2 + c + np.random.normal(0, sigma**2, size=(n,1))

X, y = my_generate_ds(n, f)

### BEGIN STUDENTS ###

# j = ...
# best_alpha = ...

# train_errors = ...
# test_errors = ...

j = 7

X_phi = Phi(X,j)

X_train = X_phi[:10, :]
X_test = X_phi[10:, :]

y_train = y[:10]
y_test = y[10:]

alpha_range = np.logspace(-10,2, 20)

test_errors = list()
train_errors = list()

for a in alpha_range:
    
    reg = Ridge(alpha=a)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    test_errors.append(mean_squared_error(y_test, y_pred))
    train_errors.append(mean_squared_error(y_train, reg.predict(X_train)))
    
### END STUDENTS ###

best_alpha = alpha_range[np.argmin(test_errors)]

reg = Ridge(alpha=best_alpha)
reg.fit(Phi(X, j), y)

### PLOTTING
fig, ax = plt.subplots()

ax.set_title("Best model")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.scatter(X, y, c=["orange"]);

# Generate points on the fitted curve
xmin = X.min()
xmax = X.max()

n_ticks = 100

ticks = np.linspace(xmin, xmax, 100).reshape((n_ticks, 1))

# Something has changed here...
y_ticks = reg.predict(Phi(ticks, j))

# Plot the fitted curves
curve1 = ax.plot(ticks, y_ticks, 'r-', label="", alpha=0.5);

fig, ax = plt.subplots(figsize=(10,5))

ax.set_title("Error curves")
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("MSE")

# Generate points on the fitted curve
xmin = X.min()
xmax = X.max()

n_ticks = 100

ticks = np.linspace(xmin, xmax, 100).reshape((n_ticks, 1))

# Something has changed here...
y_ticks = reg.predict(Phi(ticks, j))

# Plot the fitted curves
curve2 = ax.semilogx(alpha_range, np.array(train_errors), 'r-', label="Training error", alpha=0.5);
curve1 = ax.semilogx(alpha_range, np.array(test_errors), 'b-', label="Test error", alpha=0.5);

ax.vlines(best_alpha, 0,  1, 'k', linestyles='dashed', label=r"Best $\alpha$")


ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);

from sklearn.model_selection import GridSearchCV

### BEGIN STUDENTS ###

# j = ...
# k = ...
# alpha_range = ...
# grid_search = ...

j = 7
k = 3

alpha_range = np.logspace(-10,2, 20)

X_phi = Phi(X,j)

reg = Ridge()

grid_search = GridSearchCV(reg, param_grid={'alpha' : alpha_range}, cv=k)
grid_search.fit(X_phi, y)

### END STUDENTS ###


### PLOTTING
fig, ax = plt.subplots()

ax.set_title("Best model")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax.scatter(X, y, c=["orange"]);

# Generate points on the fitted curve
xmin = X.min()
xmax = X.max()

n_ticks = 100

ticks = np.linspace(xmin, xmax, 100).reshape((n_ticks, 1))

# Something has changed here...
y_ticks = grid_search.predict(Phi(ticks, j))

# Plot the fitted curves
curve1 = ax.plot(ticks, y_ticks, 'r-', label="", alpha=0.5);



