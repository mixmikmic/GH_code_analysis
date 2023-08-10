import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
get_ipython().magic('matplotlib inline')

sns.set(style="white")

def covariance_function(x, xprime, h=1.0):
    """
    A squared exponential covariance function. Since this is for
    illustration purposes, I will only allow 1D input.
    """
    n_x = len(x)
    n_xprime = len(xprime)
    
    X1 = x[:,np.newaxis].repeat(n_xprime, axis=1)
    X2 = xprime[np.newaxis].repeat(n_x, axis=0)
    
    dist = (X1-X2)**2
    return np.exp(-0.5*dist/h)

def draw_function(x, n=1, h=1.0):
    
    cov = covariance_function(x, x, h=h)
    mean = np.zeros(len(x))
    
    fvals = np.random.multivariate_normal(mean, cov, size=n)
    
    return fvals.T
    
    

x = np.linspace(-5, 5, 100)
y_free = draw_function(x, n=4, h=1)

plt.plot(x, y_free)
plt.title('Unconstrained functions with square exponential cov. function')
plt.xlabel('x')
plt.ylabel('y')

def draw_constrained_functions(x_obs, y_obs, x, n=1, h=1.0):
    
    # This implements equation 2.19 from Rasmussen & Williams (2006)
    
    # First calculate the training sample covariance matrix. We need
    # the inverse of this.
    cov_train = covariance_function(x_obs, x_obs, h=h)
    inv_cov_train = np.linalg.inv(cov_train)
    
    # The the test-test and test-training covariances
    
    cov_testtest = covariance_function(x, x, h=h)
    cov_testtrain = covariance_function(x, x_obs, h=h)
    cov_traintest = covariance_function(x_obs, x, h=h)
    
    # This matrix is needed twice so I pre-calculate it here.
    Ktmp = np.matmul(cov_testtrain, inv_cov_train)
    mean = np.matmul(Ktmp, y_obs)
    cov = cov_testtest - np.matmul(Ktmp, cov_traintest)
    
    fvals = np.random.multivariate_normal(mean, cov, size=n)
    
    return fvals.T
    

x_obs = np.array([-3, 0.5, 2.1])
y_obs = np.array([-0.5, 1, 0.9])

y = draw_constrained_functions(x_obs, y_obs, x, n=4, h=2.)

plt.plot(x, y)
plt.scatter(x_obs, y_obs, 150, 'k')

def draw_constrained_functions_w_noise(x_obs, y_obs, dy_obs, x, n=1, h=1.0):
    
    # This implements equation 2.22-2.24 from Rasmussen & Williams (2006)
    
    # First calculate the training sample covariance matrix plus the noise
    # covariance. We need the inverse of this.
    cov_noise = np.diag(dy_obs) # Assume independence
    cov_train = covariance_function(x_obs, x_obs, h=h) + cov_noise
    inv_cov_train = np.linalg.inv(cov_train)
    
    # The the test-test and test-training covariances
    
    cov_testtest = covariance_function(x, x, h=h)
    cov_testtrain = covariance_function(x, x_obs, h=h)
    cov_traintest = covariance_function(x_obs, x, h=h)
    
    # This matrix is needed twice so I pre-calculate it here.
    Ktmp = np.matmul(cov_testtrain, inv_cov_train)
    mean = np.matmul(Ktmp, y_obs)
    cov = cov_testtest - np.matmul(Ktmp, cov_traintest)
    
    fvals = np.random.multivariate_normal(mean, cov, size=n)
    
    return fvals.T, np.diag(cov)

sigma_y  = 0.05+np.zeros(len(y_obs))
y, var_y = draw_constrained_functions_w_noise(x_obs, y_obs, sigma_y, x, n=4)
ylow = np.mean(y, axis=1)-1.96*np.sqrt(var_y)
yhigh =np.mean(y, axis=1)+1.96*np.sqrt(var_y)

plt.plot(x, y)
plt.fill_between(x, ylow, yhigh, alpha=0.5, color='#cccccc')
tmp = plt.errorbar(x_obs, y_obs, sigma_y, fmt='.k', ecolor='gray', markersize=20)

def make_fake_data():
    """
    Create some data from a sine curve.
    """
    
    np.random.seed(15)
    
    n_samples = 7
    
    x = np.random.uniform(-1, 1.5, 10)*np.pi
    x.sort()
    
    y = np.sin(x)
    
    # And finally add some noise
    dy = 1.0/3.0
    y = y + np.random.normal(0, dy, len(y))
    
    return x, y, dy

def plot_a_fit(x, y, dy, xest, yest, dyest, include_true=False):
    """
    Plot the result of a fit to the fake data. This is put in as a function
    to speed things up below.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(xest, yest, '-k')
    y_low = yest - 1.96*dyest
    y_high = yest + 1.96*dyest
    ax.fill_between(xest, y_low, y_high, alpha=0.2, color='r')
    ax.errorbar(x, y, dy, fmt='.k', ecolor='gray', markersize=8)
    plt.plot(xest, yest, '-', color='#00aaff')
    
    if include_true:
        plt.plot(xest, np.sin(xest), '--', color='#999999')

from sklearn.gaussian_process import GaussianProcess

x, y, dy = make_fake_data()
xplot = np.linspace(np.min(x), np.max(x), 1000)
yplot = np.sin(xplot)

gp = GaussianProcess(corr='squared_exponential', theta0=0.1, thetaL=1e-2,
                    thetaU=1, normalize=False, nugget=(dy/y)**2, random_start=1)

g = gp.fit(x[:, np.newaxis], y)

y_pred, MSE = gp.predict(xplot[:, np.newaxis], eval_MSE=True)
sigma = np.sqrt(MSE)

plot_a_fit(x, y, dy, xplot, y_pred, sigma, include_true=True)

