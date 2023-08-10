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

def rejection_sample(p,interval_x, interval_y, n):
    xs = np.random.uniform(interval_x[0], interval_x[1],(n,))
    ys = np.random.uniform(interval_y[0], interval_y[1],(n,))
    kept  = p(xs)>ys
    return kept, np.logical_not(kept), xs, ys

def odd_pdf(x):
    return np.sin(x*5*np.pi) / ((x*5*np.pi)) + 0.22

kept, rejected, xs, ys = rejection_sample(odd_pdf, [-1,1], [0,1], 200)
plt.plot(xs[kept], ys[kept], 'r.')

plt.plot(xs[kept], np.zeros_like(xs[kept])+0.01, 'ro')
for x,y in zip(xs[kept], ys[kept]):
    plt.plot([x,x], [0.01,y], 'r', alpha=0.1)

plt.plot(xs[rejected], ys[rejected], 'k.')

xf = np.linspace(-1,1,100)
plt.plot(xf,odd_pdf(xf), '--')

print "Fraction under the curve: %.2f" % (np.sum(kept) / float(len(xs)))
        

def metropolis(p,q,x_init,n):
    x = x_init
    
    samples = []
    rejected = [] # we only keep the rejected samples to plot them later
    for i in range(n):
        # find a new candidate spot to jump to
        x_prime = x + q()
        # if it's better, go right away
        if p(x_prime)>p(x):
            x = x_prime
            samples.append(x_prime)            
        else:
            # if not, go with probability proportional to the
            # ratio between the new point and the current one
            pa = p(x_prime)/p(x)
            if np.random.uniform(0,1)<pa:
                x = x_prime
                samples.append(x_prime)
            else:
                rejected.append(x_prime)
                
    return np.array(samples), np.array(rejected)


A = np.array([[0.15, 0.9], [-0.1, 2.5]])
p = lambda x:scipy.stats.multivariate_normal(mean=[0,0], cov=A).pdf(x)
q = lambda: np.random.normal(0,0.5,(2,))
accept, reject = metropolis(p,q,[0.1, 0.3], 200)
plt.figure(figsize=(10,10))
plt.plot(accept[:,0], accept[:,1])
plt.plot(accept[:,0], accept[:,1], 'b.')
plt.plot(reject[:,0], reject[:,1], 'r.')
x,y = np.meshgrid(np.linspace(-5,5,30), np.linspace(-4,4,30))
plt.imshow(p(np.dstack([x,y])), extent=[-4,4,-4,4], cmap='viridis')
plt.grid("off")
plt.title("MCMC sampling with Metropolis-Hastings")        

# run the chain for longer, and plot the trace and the histogram of the variable
accept, reject = metropolis(p,q,[0.1, 0.3], 2000)
plt.subplot(1,2,1)
plt.plot(accept[:,0])
plt.title("Trace for $x$")
plt.subplot(1,2,2)
plt.hist(accept[:,0], bins=20)
plt.title("Histogram of $x$")

plt.figure()

plt.plot(accept[:,1])
plt.subplot(1,2,1)
plt.title("Trace for $y$")
plt.plot(accept[:,0])
plt.subplot(1,2,2)
plt.title("Histogram of $y$")
plt.hist(accept[:,0], bins=20);

## Burn-in and thinning plot
y = np.random.normal(0,1,(500,))

# introduce correlations
y = scipy.signal.lfilter([1,1, 1, 1], [1], y)
x = np.arange(len(y))

burn = 100
thin = 4
plt.plot(x[0:burn], y[0:burn], 'r:')
plt.plot(x[burn::thin], y[burn::thin], 'ko')
plt.plot(x[burn:], y[burn:], 'g:')
plt.plot(x[burn:], y[burn:], 'g.')


plt.axvline(burn, c='r')
plt.text(15,2.5,"Burn-in period")

def numerify(string):
    # remove all but letters and space (note that this is not a very efficient way to do this process)
    # and then convert to 0=space, 1-27 = a-z
    filtered_string = [max(1+ord(c.lower()) - ord('a'), 0) for c in string if c.isalpha() or c.isspace()]
    return filtered_string

def learn_bigram(string):
    # return a matrix with the bigram counts from string, including only letters and whitespace
    coded = numerify(string)
    joint = np.zeros((27,27))
    # iterate over sequential pairs
    for prev, this in zip(coded[:-1], coded[1:]):
        joint[prev, this] += 1
    # note that we add on an epsilon to avoid dividing by zero!
    bigram = joint.T / (np.sum(joint, axis=0)+1e-6)
    return bigram.T, joint.T
        
    

with open("data/macbeth.txt") as f:
    macbeth_bigram, macbeth_joint = learn_bigram(f.read())

with open("data/metamorphosis.txt") as f:
    metamorphosis_bigram, metamorphosis_joint = learn_bigram(f.read())
    

# The joint distribution
plt.imshow(macbeth_joint, interpolation='nearest', cmap="viridis")
plt.xticks(range(27), ' abcdefghijklmnopqrstuvwxyz')
plt.yticks(range(27), ' abcdefghijklmnopqrstuvwxyz')
plt.grid('off')

# The conditional distributions
plt.imshow(macbeth_bigram, interpolation='nearest', cmap="viridis")
plt.xticks(range(27), ' abcdefghijklmnopqrstuvwxyz')
plt.yticks(range(27), ' abcdefghijklmnopqrstuvwxyz')
plt.grid('off')

plt.figure()
plt.imshow(metamorphosis_bigram, interpolation='nearest', cmap="viridis")
plt.xticks(range(27), ' abcdefghijklmnopqrstuvwxyz')
plt.yticks(range(27), ' abcdefghijklmnopqrstuvwxyz')
plt.grid('off')

def log_likelihood_bigram(string, bigram):
   symbols = numerify(string)
   llik = 0
   # we sum the log probabilities to avoid numerical underflow
   for prev, this in zip(symbols[:-1], symbols[1:]):
       llik += np.log(bigram[prev, this]+1e-8) 
   return llik


def compare_logp(fname):
    print fname

    with open(fname) as f:
        text = f.read()
        mb_llik = log_likelihood_bigram(text, macbeth_bigram) / len(text)
        mm_llik = log_likelihood_bigram(text, metamorphosis_bigram) / len(text)
        diff = mb_llik - mm_llik
        print "\tlogp Macbeth: % 10.2f\t logp Metamorphosis:% 10.2f\t Difference logp:% 10.2f" % (mb_llik, mm_llik, diff)
        if diff>0:
            print "\tModel favours: Macbeth"            
        else:
            print "\tModel favours: Metamorphosis"

    print
    
compare_logp("data/macbeth.txt")
compare_logp("data/metamorphosis.txt")
compare_logp("data/romeo_juliet.txt")
compare_logp("data/the_trial.txt")


## Example, using the linear regression model from the last section:
import numpy.ma as ma # masked array support


## generate the data for the regression
x = np.sort(np.random.uniform(0,20, (50,)))
m = 2
c = 15
# Add on some measurement noise, with std. dev. 3.0
epsilon = data = np.random.normal(0,3, x.shape)
y = m * x + c + epsilon

## Now the imputation; we will try and infer missing some missing values of y (we still have the corresponding x)
## mark last three values of y invalid
y_impute = y[:]
y_impute[-3:] = 0
y_impute = ma.masked_equal(y_impute,0)
print "Y masked for imputation:", y_impute # we will see the last three entries with --

# create the model (exactly as before, except we switch "y_impute" for "y")
m_unknown = mc.Normal('m', 0, 0.01)
c_unknown = mc.Normal('c', 0, 0.001)
precision = mc.Uniform('precision', lower=0.001, upper=10.0)
x_obs = mc.Normal("x_obs", 0, 1, value=x, observed=True)
@mc.deterministic(plot=False)
def line(m=m_unknown, c=c_unknown, x=x_obs):
    return x*m+c
y_obs =  mc.Normal('y_obs', mu=line, tau=precision, value=y_impute, observed=True)
model = mc.Model([m_unknown, c_unknown, precision, x_obs, y_obs])

# sample from the distribution
mcmc = mc.MCMC(model)
mcmc.sample(iter=10000, burn=2000, thin=5)

## now we will have three entries in the y_obs trace from this run
y_trace = mcmc.trace('y_obs')[:]

## the original data
plt.plot(x[:-3],y[:-3], '.')
plt.plot(x[-3:],y[-3:], 'go')
plt.plot(x,x*m+c, '-')

# samples from posterior predicted for the missing values of y
plt.plot(np.tile(x[-3], (len(y_trace[:,0]), 1)), y_trace[:,0],  'r.', alpha=0.01)
plt.plot(np.tile(x[-2], (len(y_trace[:,1]), 1)), y_trace[:,1],  'r.', alpha=0.01)
plt.plot(np.tile(x[-1], (len(y_trace[:,2]), 1)), y_trace[:,2],  'r.', alpha=0.01)


# Solution

## the variables in the model should go in the list passed to Model
model = pymc.Model([])

## see the graphical representation of the model
show_dag(model)

## Construct a sampler
mcmc = pymc.MCMC(model)

## Sample from the result; you should try changing the number of iterations
mcmc.sample(iter=100000, burn=2000, thin=5)

## Use the trace methods from pymc to explore the distribution of values

## Plot the predictive distributions

