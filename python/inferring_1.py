import numpy as np
import sklearn.datasets, sklearn.linear_model, sklearn.neighbors
import sklearn.manifold, sklearn.cluster
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os, time
import scipy.io.wavfile, scipy.signal
import pymc as mc
import cv2
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (18.0, 10.0)

get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')

def joint_marginal(cov):
    # create an independent 2D normal distribution
    x,y = np.meshgrid(np.linspace(-3,3,50), np.linspace(-3,3,50))
    pos = np.empty(x.shape + (2,))
    pos[:,:,0] = x
    pos[:,:,1] = y
    joint_pdf = scipy.stats.multivariate_normal.pdf(pos, [0,0], cov)
    fig = plt.figure()
    # plot the joint
    ax = fig.add_subplot(2,2,1)
    ax.axis('equal')
    plt.title("Joint p(x,y)")
    ax.pcolor(x,y,joint_pdf, cmap='viridis')
    # plot the marginals
    ax = fig.add_subplot(2,2,3)
    ax.axis('equal')
    plt.title("Marginal $p(x) = \int\  p(x,y) dy$")
    ax.plot(x[0,:], np.sum(joint_pdf, axis=0))
    ax = fig.add_subplot(2,2,2)
    ax.axis('equal')
    plt.title("Marginal $p(y) = \int\  p(x,y) dx$")
    ax.plot(np.sum(joint_pdf, axis=1), x[0,:])
    # plot p(x|y)
    ax = fig.add_subplot(2,2,4)
    ax.axis('equal')
    plt.title("Conditional $p(x|y) = \\frac{p(x,y)}{p(x)}$")
    marginal = np.tile(np.sum(joint_pdf, axis=0), (joint_pdf.shape[0],1))
    ax.pcolor(x,y,joint_pdf/marginal, cmap='viridis')
joint_marginal([[1,0],[0.5,1]])

def prior_posterior():
    mean = 0
    std = 1
    prior = scipy.stats.norm(mean,std)
    evidence = scipy.stats.norm(1, 0.1)
    xs = np.linspace(-5,5,200)
    plt.plot(xs, prior.pdf(xs), label="Prior")
    for i in range(10):
        # this is **not** inference! just a visual example!
        mean = 0.8*mean + 0.2*1
        std = 0.8*std + 0.2*0.05
        v = evidence.rvs()
        plt.plot([v,v],[0,1], 'c', alpha=0.7)
        plt.plot(xs, scipy.stats.norm(mean,std).pdf(xs), 'k:', alpha=0.5)
    plt.plot([v,v],[0,1], 'c', alpha=0.7, label="Observations")
        
    plt.plot(xs, scipy.stats.norm(mean,std).pdf(xs), 'g', label="Posterior")
    plt.legend()
    
prior_posterior()

### Bayesian Linear Regression with pymc
### We use Monte Carlo sampling to estimate the distribution of a linear function with a normally
### distributed error, given some observed data.
### Vaguely based on: http://matpalm.com/blog/2012/12/27/dead_simple_pymc/ and http://sabermetricinsights.blogspot.co.uk/2014/05/bayesian-linear-regression-with-pymc.html

## Utility function to plot the graph of a PyMC model
def show_dag(model):
    dag = mc.graph.dag(model)
    dag.write("graph.png",format="png")
    from IPython.display import Image
    i = Image(filename='graph.png')
    return i

## generate data with a known distribution
## this will be our "observed" data
x = np.sort(np.random.uniform(0,20, (50,)))
m = 2
c = 15

# Add on some measurement noise, with std. dev. 3.0
epsilon = data = np.random.normal(0,3, x.shape)
y = m * x + c + epsilon

plt.plot(x,y, '.', label="Datapoints")
plt.plot(x, m*x+c, '--', lw=3, label="True")
plt.legend()
plt.xlabel("x")
plt.xlabel("y")

## Now, set up the PyMC model
## specify the prior distribution of the unknown line function variables
## Here, we assume a normal distribution over m and c
m_unknown = mc.Normal('m', 0, 0.01)
c_unknown = mc.Normal('c', 0, 0.001)

## specify a prior over the precision (inverse variance) of the error term
# precision = 1/variance
## Here we specify a uniform distribution from 0.001 to 10.0
precision = mc.Uniform('precision', lower=0.001, upper=10.0)

# specify the observed input variable
# we use a normal distribution, but this has no effect -- the values are fixed and the paramters
# never updated; this is just a way of transforming x into a variable pymc can work with
# (it's really a hack)
x_obs = mc.Normal("x_obs", 0, 1, value=x, observed=True)

@mc.deterministic(plot=False)
def line(m=m_unknown, c=c_unknown, x=x_obs):
    return x*m+c

# specify the observed output variable (note if use tau instead of sigma, we use the precision paramterisation)
y_obs =  mc.Normal('y_obs', mu=line, tau=precision, value=y, observed=True)

model = mc.Model([m_unknown, c_unknown, precision, x_obs, y_obs])

# display the graphical model
show_dag(model)

# sample from the distribution
mcmc = mc.MCMC(model)
mcmc.sample(iter=10000)

## plot histograms of possible parameter values
plt.figure()
plt.hist(mcmc.trace("m")[:], normed=True, bins=30)
plt.title("Estimate of m")
plt.figure()
plt.hist(mcmc.trace("c")[:], normed=True, bins=30)
plt.title("Estimate of c")
plt.figure()
plt.hist(np.sqrt(1.0/mcmc.trace("precision")[:]), normed=True, bins=30)
plt.title("Estimate of epsilon std.dev.")
plt.figure()

## now plot overlaid samples from the linear function
ms = mcmc.trace("m")[:]
cs = mcmc.trace("c")[:]

plt.title("Sampled fits")
plt.plot(x, y, '.', label="Observed")
plt.plot(x, x*m+c, '--', label="True")
xf = np.linspace(-20,40,200)
for m,c in zip(ms[::20], cs[::20]):    
    plt.plot(xf, xf*m+c, 'r-', alpha=0.005)
plt.legend()
plt.xlim(-20,40)
plt.ylim(-40,80)

## Adapted from the example given at 
## http://stackoverflow.com/questions/18987697/how-to-model-a-mixture-of-3-normals-in-pymc

n = 3
ndata = 500

## A Dirichlet model specifies the distribution over categories
## All 1 means that every category is equally likely
dd = mc.Dirichlet('dd', theta=(1,)*n)

## This variable "selects" the category (i.e. the normal distribution)
## to use. The Dirichlet distribution sets the prior over the categories.
category = mc.Categorical('category', p=dd, size=ndata)

## Now we set our priors the precision and mean of each normal distribution
## Note the use of "size" to generate a **vector** of variables (i.e. one for each category)

## We expect the precision of each normal to be Gamma distributed (this mainly forces it to be positive!)
precs = mc.Gamma('precs', alpha=0.1, beta=0.1, size=n)

## And the means of the normal to be normally distributed, with a precision of 0.001 (i.e. std. dev 1000)
means = mc.Normal('means', 0, 0.001, size=n)

## These deterministic functions link the means of the observed distribution to the categories
## They just select one of the elements of the mean/precision vector, given the current value of category
## The input variables must be specified in the parameters, so that PyMC knows which variables to pass to it
@mc.deterministic
def mean(category=category, means=means):
    return means[category]

@mc.deterministic
def prec(category=category, precs=precs):
    return precs[category]

## Generate synthetic mixture-of-normals data, with means at -50,0,+50, and std. dev of 1
v = np.random.randint( 0, n, ndata)
data = (v==0)*(np.random.normal(50,5,ndata)) + (v==1)*(np.random.normal(-50,5,ndata)) + (v==2)*np.random.normal(0,5,ndata)


## Plot the original data
plt.hist(data, bins=50)  

## Now we specify the variable we observe -- which is normally distributed, *but*
## we don't know the mean or precision. Instead, we pass the **functions** mean() and pred()
## which will be used at each sampling step.
## We specify the observed values of this node, and tell PyMC these are observed 
## This is all that is needed to specify the model
obs = mc.Normal('obs', mean, prec, value=data, observed = True)

## Now we just bundle all the variables together for PyMC
model = mc.Model({'dd': dd,
              'category': category,
              'precs': precs,
              'means': means,
              'obs': obs})


def show_dag(model):
    dag = mc.graph.dag(model)
    dag.write("graph.png",format="png")
    from IPython.display import Image
    i = Image(filename='graph.png')
    return i
    
show_dag(model)    

mcmc = mc.MCMC(model)

## Now we tell the sampler what method to use
## Metropolis works well, but we must tell PyMC to use a specific
## discrete sampler for the category variable to get good results in a reasonable time
mcmc.use_step_method(mc.AdaptiveMetropolis, model.means)
mcmc.use_step_method(mc.AdaptiveMetropolis, model.precs)
mcmc.use_step_method(mc.DiscreteMetropolis, model.category) ## this step is key!
mcmc.use_step_method(mc.AdaptiveMetropolis, model.dd)

## Run the sampler
mcmc.sample(iter=125000, burn=2000)

plt.figure()
plt.hist(mcmc.trace('means').gettrace()[:], normed=True)
plt.title("Estimated means")
plt.legend(['Component 1', 'Component 2', 'Component 3'])
plt.figure()
## show the result in terms of std. dev. (i.e sqrt(1.0/precision))
plt.title("Estimated std. dev")
plt.hist(np.sqrt(1.0/mcmc.trace('precs').gettrace()[:]), normed=True)
plt.legend(['Component 1', 'Component 2', 'Component 3'])

## the variables in the model should go in the list passed to Model
model = mc.Model([])

## see the graphical representation of the model
show_dag(model)

## Construct a sampler
mcmc = mc.MCMC(model)

## Sample from the result; you should try changing the number of iterations
mcmc.sample(iter=10000)

## Use the trace methods from pymc to explore the distribution of values

