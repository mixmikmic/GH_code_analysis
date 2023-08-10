#%matplotlib inline

import numpy as np
import pylab as pl
from scipy import linalg as sl

def cov_kernel(x1,x2,h,lam):
    """
    Squared-Exponential covariance kernel
    """
    k12 = h**2*np.exp(-1.*(x1 - x2)**2/lam**2)
    
    return k12

def make_K(x, h, lam):
    
    """
    Make covariance matrix from covariance kernel
    """
    
    # for a data array of length x, make a covariance matrix x*x:
    K = np.zeros((len(x),len(x)))
    
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            
            if (i==j):
                noise = 1e-5
            else:
                noise = 0.0
            
            # calculate value of K for each separation:
            K[i,j] = cov_kernel(x[i],x[j],h,lam) + noise**2
            
    return K

# make an array of 200 evenly spaced positions between 0 and 20:
x1 = np.arange(0, 20.,0.01)
    
for i in range(0,3):
    
    h = 1.0
    
    if (i==0): lam = 0.1
    if (i==1): lam = 1.0
    if (i==2): lam = 5.0
        
    # make a covariance matrix:
    K = make_K(x1,h,lam)
    
    # five realisations:
    for j in range(0,5):
        
        # draw samples from a co-variate Gaussian distribution, N(0,K):
        y1 = np.random.multivariate_normal(np.zeros(len(x1)),K)
    
        tmp2 = '23'+str(i+3+1)
        pl.subplot(int(tmp2))
        pl.plot(x1,y1)
        
        
    tmp1 = '23'+str(i+1)
    pl.subplot(int(tmp1))
    pl.imshow(K)
    pl.title(r"$\lambda = $"+str(lam))
    
    
pl.show()

# set number of training points
nx_training = 5

# randomly select the training points:
tmp = np.random.uniform(low=0.0, high=2000.0, size=nx_training)
tmp = tmp.astype(int)

condition = np.zeros_like(x1)
for i in tmp: condition[i] = 1.0
    
y_train = y1[np.where(condition==1.0)]
x_train = x1[np.where(condition==1.0)]
y_test = y1[np.where(condition==0.0)]
x_test = x1[np.where(condition==0.0)]

# define the covariance matrix:
K = make_K(x_train,h,lam)

# take the inverse:
iK = np.linalg.inv(K)

print 'determinant: ',np.linalg.det(K), sl.det(K)

mu=[];sig=[]
for xx in x_test:

    # find the 1d covariance matrix:
    K_x = cov_kernel(xx, x_train, h, lam)
    
    # find the kernel for (x,x):
    k_xx = cov_kernel(xx, xx, h, lam)
    
    # calculate the posterior mean and variance:
    mu_xx = np.dot(K_x.T,np.dot(iK,y_train))
    sig_xx = k_xx - np.dot(K_x.T,np.dot(iK,K_x))
    
    mu.append(mu_xx)
    sig.append(np.sqrt(np.abs(sig_xx))) # note sqrt to get stdev from variance

# mu and sig are currently lists - turn them into numpy arrays:
mu=np.array(mu);sig=np.array(sig)

# make some plots:

# left-hand plot
ax = pl.subplot(121)
pl.scatter(x_train,y_train)  # plot the training points
pl.plot(x1,y1,ls=':')        # plot the original data they were drawn from
pl.title("Input")

# right-hand plot
ax = pl.subplot(122)
pl.plot(x_test,mu,ls='-')     # plot the predicted values
pl.plot(x_test,y_test,ls=':') # plot the original values


# shade in the area inside a one standard deviation bound:
ax.fill_between(x_test,mu-sig,mu+sig,facecolor='lightgrey', lw=0, interpolate=True)
pl.title("Predicted")

pl.scatter(x_train,y_train)  # plot the training points

# display the plot:
pl.show()

# set number of training points
frac = 0.05
nx_training = int(len(x1)*frac)
print "Using ",nx_training," points."

# randomly select the training points:
tmp = np.random.uniform(low=0.0, high=2000.0, size=nx_training)
tmp = tmp.astype(int)

condition = np.zeros_like(x1)
for i in tmp: condition[i] = 1.0
    
y_train = y1[np.where(condition==1.0)]
x_train = x1[np.where(condition==1.0)]
y_test = y1[np.where(condition==0.0)]
x_test = x1[np.where(condition==0.0)]

pl.scatter(x_train,y_train)  # plot the training points
pl.plot(x1,y1,ls=':')        # plot the original data they were drawn from
pl.title("Training Data")
pl.show()

import george
from george import kernels

import emcee

# we need to define three functions: 
# a log likelihood, a log prior & a log posterior.

# set the loglikelihood:
def lnlike2(p, x, y):
    
    # update kernel parameters:
    gp.kernel[:] = p
    
    # compute covariance matrix:
    gp.compute(x)
    
    # calculate the likelihood:
    ll = gp.lnlikelihood(y, quiet=True)
    
    # return 
    return ll if np.isfinite(ll) else 1e25

# set the logprior
def lnprior2(p):
    
    # note that "p" contains the ln()
    # of the parameter values - set your
    # prior ranges appropriately!
    
    lnh,lnlam,lnsig = p
    
    # really crappy prior:
    if (-2.<lnh<5.) and (-1.<lnlam<10.) and (-30.<lnsig<0.):
        return 0.0
    
    return -np.inf

# set the logposterior:
def lnprob2(p, x, y):
    
    lp = lnprior2(p)
    
    return lp + lnlike2(p, x, y) if np.isfinite(lp) else -np.inf


# initiate george with the exponential squared kernel:
kernel = 1.0*kernels.ExpSquaredKernel(30.0)+kernels.WhiteKernel(1e-5)
gp = george.GP(kernel, mean=0.0)

# put all the data into a single array:
data = (x_train,y_train)

# set your initial guess parameters
# remember george keeps these in ln() form!
initial = gp.kernel[:]
print "Initial guesses: ",np.exp(initial)

# set the dimension of the prior volume 
# (i.e. how many parameters do you have?)
ndim = len(initial)
print "Number of parameters: ",ndim

# The number of walkers needs to be more than twice 
# the dimension of your parameter space... unless you're crazy!
nwalkers = 10

# perturb your inital guess parameters very slightly
# to get your starting values:
p0 = [np.array(initial) + 1e-4 * np.random.randn(ndim)
      for i in xrange(nwalkers)]

# initalise the sampler:
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=data)

# run a few samples as a burn-in:
print("Running burn-in")
p0, lnp, _ = sampler.run_mcmc(p0, 1000)
sampler.reset()

# take the highest likelihood point from the burn-in as a
# starting point and now begin your production run:
print("Running production")
p = p0[np.argmax(lnp)]
p0 = [p + 1e-4 * np.random.randn(ndim) for i in xrange(nwalkers)]
p0, _, _ = sampler.run_mcmc(p0, 8000)

print "Finished"

import acor

# calculate the convergence time of our
# MCMC chains:
samples = sampler.flatchain
s2 = np.ndarray.transpose(samples)
tau, mean, sigma = acor.acor(s2)
print "Convergence time from acor: ", tau

# get rid of the samples that were taken
# before convergence:
delta = int(20*tau)
samples = sampler.flatchain[delta:,:]

# extract the log likelihood for each sample:
lnl = sampler.flatlnprobability[delta:]

# find the point of maximum likelihood:
ml = samples[np.argmax(lnl),:]

# print out those values
# Note that we have unwrapped 
print "ML parameters: ", 
print "h: ", np.sqrt(np.exp(ml[0]))," ; lam: ",np.sqrt(np.exp(ml[1]))," ; sigma: ",np.sqrt(np.exp(ml[2]))

import corner

# Plot it.
figure = corner.corner(samples, labels=[r"$\ln(h^2)$", r"$\ln(\lambda^2)$", r"$\ln(\sigma^2)$"],
                         truths=ml,
                         quantiles=[0.16,0.5,0.84],
                         #levels=[0.39,0.86,0.99],
                         levels=[0.68,0.95,0.99],
                         title="Mauna Loa Data",
                         show_titles=True, title_args={"fontsize": 12})

q1 = corner.quantile(samples[:,0],
						 q=[0.16,0.5,0.84])

print "Parameter 1: ",q1[1],"(-",q1[1]-q1[0],", +",q1[2]-q1[1],")"

q2 = corner.quantile(samples[:,1],
						 q=[0.16,0.5,0.84])

print "Parameter 2: ",q2[1],"(-",q2[1]-q2[0],", +",q2[2]-q2[1],")"

q3 = corner.quantile(samples[:,2],
						 q=[0.16,0.5,0.84])

print "Parameter 3: ",q2[1],"(-",q2[1]-q2[0],", +",q2[2]-q2[1],")"




