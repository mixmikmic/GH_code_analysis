import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns # Comment this out if you don't have it
sns.set_style('white')
sns.set_context('talk')
import GPy
# The input dimension
dim = 1
# The variance of the covariance kernel
variance = 1.
# The lengthscale of the covariance kernel
ell = 0.3
# Generate the covariance object
k = GPy.kern.RBF(dim, variance=variance, lengthscale=ell)
# Print it
print k
# and plot it
k.plot()

from ipywidgets import interactive
def plot_kernel(variance=1., ell=0.3):
    k = GPy.kern.RBF(dim, variance=variance, lengthscale=ell)
    k.plot()
    plt.ylim(0, 10)
interactive(plot_kernel, variance=(1e-3, 10., 0.01), ell=(1e-3, 10., 0.01))

from ipywidgets import interactive
def plot_kernel(variance=1., ell1=0.3, ell2=0.3):
    k = GPy.kern.RBF(2, ARD=True, variance=variance,
                     lengthscale=[ell1, ell2])  # Notice that I just changed the dimension here
    k.plot()
interactive(plot_kernel, variance=(1e-3, 10., 0.01), ell1=(1e-3, 10., 0.01), ell2=(1e-3, 10., 0.01))

# Number of dimensions
dim = 1

# Number of input points
n = 20

# The lengthscale
ell = .1

# The variance 
variance = 1.

# The covariance function
k1 = GPy.kern.RBF(dim, lengthscale=ell, variance=variance)

# Draw a random set of inputs points in [0, 1]^dim
X = np.random.rand(n, dim)

# Evaluate the covariance matrix on these points
K = k1.K(X)

# Compute the eigenvalues of this matrix
eig_val, eig_vec = np.linalg.eigh(K)

# Plot the eigenvalues (they should all be positive)
print '> plotting eigenvalues of K'
print '> they must all be positive'
fig, ax = plt.subplots()
ax.plot(np.arange(1, n+1), eig_val, '.')
ax.set_xlabel('$i$', fontsize=16)
ax.set_ylabel('$\lambda_i$', fontsize=16)

# Now create another (arbitrary) covariance function
k2 = GPy.kern.Exponential(dim, lengthscale=0.2, variance=2.1)

# Create a new covariance function that is the sum of these two:
k_new = k1 + k2

# Let's plot the new covariance
fig, ax = plt.subplots()
k1.plot(ax=ax, label='$k_1$')
k2.plot(ax=ax, label='$k_2$')
k_new.plot(ax=ax, label='$k_1 + k_2$')
plt.legend(fontsize=16);

# If this is a valid covariance function, then it must 
# be positive definite
# Compute the covariance matrix:
K_new = k_new.K(X)

# and its eigenvalues
eig_val_new, eig_vec_new = np.linalg.eigh(K_new)

# Plot the eigenvalues (they should all be positive)
print '> plotting eigenvalues of K'
print '> they must all be positive'
fig, ax = plt.subplots()
ax.plot(np.arange(1, n+1), eig_val_new, '.')
ax.set_xlabel('$i$', fontsize=16)
ax.set_ylabel('$\lambda_i$', fontsize=16);

# To gaurantee reproducibility
np.random.seed(123456)

# Number of test points
num_test = 10

# Pick a covariance function
k = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=.1)

# Pick a mean function
mean_func = lambda(x): np.zeros(x.shape)

# Pick a bunch of points over which you want to sample the GP
X = np.linspace(0, 1, num_test)[:, None]

# Evaluate the mean function at X
m = mean_func(X)

# Compute the covariance function at these points
nugget = 1e-6 # This is a small number required for stability
C = k.K(X) + nugget * np.eye(X.shape[0])

# Compute the Cholesky of the covariance
# Notice that we need to do this only once
L = np.linalg.cholesky(C)

# Number of samples to take
num_samples = 3

# Take 3 samples from the GP and plot them:
fig, ax = plt.subplots()
# Plot the mean function
ax.plot(X, m)
for i in xrange(num_samples):
    z = np.random.randn(X.shape[0], 1)    # Draw from standard normal
    f = m + np.dot(L, z)                  # f = m + L * z
    ax.plot(X, f, color=sns.color_palette()[1], linewidth=1)
#ax.set_ylim(-6., 6.)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$y$', fontsize=16)
ax.set_ylim(-5, 5);

