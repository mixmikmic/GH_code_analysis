get_ipython().magic('pylab inline')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#create some data and adding a white noise to the data
x = np.linspace(-10, 10, 100)
A = 2
B = 3
#We use the following equation to generate our data
y = A*np.arctan(x/B)

#adding the noise 
noise = np.random.normal(0,0.1,len(y))
y_observe = y + noise

plt.plot(x, y_observe, 'o')
plt.xlabel('X', fontsize = 16)
plt.ylabel('Y', fontsize = 16)
plt.show()

A_space = np.arange(1, 10.25, 0.25)
B_space = np.arange(1, 10.25, 0.25)
residual_mat = np.zeros((len(A_space), len(B_space)))
#loop through parameter space, assume we know that A and B are ranging from 1 - 9
for i, A in enumerate(A_space):
    for j, B in enumerate(B_space):
        #estimate y
        y_est = A*np.arctan(x/B)
        sum_residual = sum((y_est - y_observe)**2)
        residual_mat[i, j] = sum_residual
        
#get the index of the smallest residual of all the models
ix_min = np.where(residual_mat == np.min(residual_mat))
A_est, B_est= A_space[ix_min[0][0]], B_space[ix_min[1][0]]
#plot the fitting line on the observed data
plt.plot(x, y_observe, 'o')
plt.plot(x, A_est*np.arctan(x/B_est), 'r', label = 'fitted line')
plt.xlabel('X', fontsize = 16)
plt.ylabel('Y', fontsize = 16)
plt.legend(fontsize = 16, loc = 2)
plt.title('The best model with parameter\n A = %.2f, and B = %.2f' %(A_est, B_est))
plt.show()

#visualize the space of the sum of squared residual
fig = plt.figure()
ax = Axes3D(fig)
X, Y = np.meshgrid(A_space, B_space)
ax.plot_surface(X, Y, residual_mat, rstride=1, cstride=1, cmap=cm.jet)
ax.set_xlabel('A', fontsize = 16)
ax.set_ylabel('B', fontsize = 16)
ax.set_zlabel('Residual^2 sum', fontsize = 16)
plt.title('The space of the sum of squared residual with minimum at\n A = %.2f, and B = %.2f' %(A_est, B_est))
plt.show()

from pymc import Model, Normal, HalfNormal
import pymc as pm
import corner

def myModel(x, y_observe): 
    # Priors for unknown model parameters
    #Note tau is the 1/sigma^2, e.g 1/σ^2
    A_prior = pm.distributions.Normal('A', mu = 2, tau = 100)
    B_prior = pm.distributions.Normal('B', mu = 3, tau = 100)
    
    #You can try different distributions on the variance of the noise
    #sigma = basic_model.HalfNormal('sigma', 1)
    #tau_prior = pm.distributions.Gamma("tau", alpha=0.001, beta=0.001)
    #Here I use sigma = 0.1
    tau_prior = 1.0/0.1**2
    
    # Expected value of outcome
    @pm.deterministic(plot=False)
    def f(x = x, A=A_prior, B=B_prior):
        return A*np.arctan(x/B)
    
    # Likelihood (sampling distribution) of observations
    Y_obs = pm.distributions.Normal('Y_obs', mu = f, tau = tau_prior, value = y_observe, observed=True)
    return locals()

#Here I also calculate MAP to get the best solution
model = pm.Model(myModel(x, y_observe))
map_ = pm.MAP(model)
map_.fit()
#print the MAP solution
print(map_.A_prior.value, map_.B_prior.value)

#Use MCMC to get the distribution of parameters
mcmc = pm.MCMC(myModel(x, y_observe))
mcmc.sample(iter=50000, burn=20000)
pm.Matplot.plot(mcmc)
#The trace is the series of the randm walk for each parameter, what we expect from this figure is no obvious trend. If the beginning of the run looks very different from the end, then the simulated Markov Chain is nowhere near stationarity and we need run longer.
#The Autocorrelation plot shows the cross-correlation of a signal with itself at different points in time. The x axis is time lag, and y is autocorrelation coefficient. What we are looking for is what lag the autocorrelations decrease to being not significantly different from zero. If autocorrelation is high, you will have to use a longer burn-in, and you will have to draw more samples after the burn-in to get a good estimate of the posterior distribution..Low autocorrelation means good exploration. Exploration and convergence are essentially the same thing. If a sampler explores well, then it will converge fast, and vice versa. Part of the confusion here is that the book does not explain the concept of convergence very well.
#Reference: http://stats.stackexchange.com/questions/116861/posterior-autocorrelation-in-pymc-how-to-interpret-it

print(mcmc.A_prior.summary())
print(mcmc.B_prior.summary())

#plot the fitting
plt.plot(x, y_observe, 'o')
plt.plot(x, np.mean(mcmc.A_prior.trace())*np.arctan(x/np.mean(mcmc.B_prior.trace())), 'r', label = 'fitted line')
plt.xlabel('X', fontsize = 16)
plt.ylabel('Y', fontsize = 16)
plt.legend(fontsize = 16, loc = 2)
plt.show()

#let's plot the best solution with the true solution, you need corner as dependencies, e.g. sudo pip install corner
samples = array([mcmc.A_prior.trace(),mcmc.B_prior.trace()]).T
tmp = corner.corner(samples[:,:], labels=['A','B'], 
                truths=[2, 3])

