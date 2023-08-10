# This is the first cell with code: set up the Python environment
get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import division, print_function
import matplotlib.pyplot as plt
import math
import numpy as np
import numpy.random
import scipy as sp
import scipy.stats
# For interactive widgets
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import clear_output, display, HTML

np.random.seed(1234567890) # set seed for reproducibility

def LRFromTrials(trials, N, p0, p1):
    '''
       Finds the sequence of likelihood ratios for the hypothesis that the population 
       percentage is p1 to the hypothesis that it is p0, for sampling without replacement
       from a population of size N.
    '''
    A = np.cumsum(np.insert(trials, 0, 0)) # so that cumsum does the right thing
    terms = np.ones(N)
    for k in range(len(trials)):
        if trials[k] == 1.0:
            if (N*p0 - A[k]) > 0:
                terms[k] = np.max([N*p1 - A[k], 0])/(N*p0 - A[k])
            else:
                terms[k] = np.inf
        else:
            if (N*(1-p0) - k + 1 + A[k]) > 0:
                terms[k] = np.max([(N*(1-p1) - k + 1 + A[k]), 0])/(N*(1-p0) - k + 1 + A[k])
            else:
                terms[k] = np.inf
    return(np.cumprod(terms))

def plotBernoulliSPRT(N, p, p0, p1, alpha):
    '''
       Plots the progress of a one-sided SPRT for N dependent Bernoulli trials 
       in sampling without replacement from a population of size N with a 
       fraction p of items labeled "1," for testing the hypothesis that p=p0 
       against the hypothesis p=p1 at significance level alpha
    '''
    plt.clf()
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    trials = np.zeros(N)
    nOnes = int(math.floor(N*p))
    trials[0:nOnes] = np.ones(nOnes)
    np.random.shuffle(trials) # items are in random order

    LRs = np.nan_to_num(LRFromTrials(trials, N, p0, p1))
    logLRs = np.nan_to_num(np.log(LRs))
    
    LRs[LRs > 10**6] = 10**6 # avoid plot overflow
    logLRs[logLRs > 10**6] = 10**6 # avoid plot overflow
    
    #
    ax[0].plot(range(N),LRs, color='b')
    ax[0].axhline(y=1/alpha, xmin=0, xmax=N, color='g', label=r'$1/\alpha$')
    ax[0].set_title('Simulation of Wald\'s SPRT for population percentage, w/o replacement')
    ax[0].set_ylabel('LR')
    ax[0].legend(loc='best')
    #
    ax[1].plot(range(N),logLRs, color='b', linestyle='--')
    ax[1].axhline(y=math.log(1/alpha), xmin=0, xmax=N, color='g', label=r'$log(1/\alpha)$')
    ax[1].set_ylabel('log(LR)')
    ax[1].set_xlabel('trials')
    ax[1].legend(loc='best')
    plt.show()


interact(plotBernoulliSPRT,         N=widgets.IntSlider(min=500, max=50000, step=500, value=5000),         p=widgets.FloatSlider(min=0.001, max=1, step=0.01, value=.51),         p0=widgets.FloatSlider(min=0.001, max=1, step=0.01, value=.5),         p1=widgets.FloatSlider(min=0.001, max=1, step=0.01, value=.51),         alpha=widgets.FloatSlider(min=0.001, max=1, step=0.01, value=.05)
        )

alpha = 0.05                   # significance level
reps = int(10**4)              # number of replications
p, p0, p1 = [0.525, 0.5, 0.525]  # need p > p0 or might never reject
N = 10000                       # population size
dist = np.zeros(reps)          # allocate space for the results

trials = np.zeros(N)
nOnes = int(math.floor(N*p))
trials[0:nOnes] = np.ones(nOnes) # trials now contains math.floor(n*p) ones

for i in np.arange(reps):
    np.random.shuffle(trials) # items are in random order
    LRs = LRFromTrials(trials, N, p0, p1) # likelihood ratios for this realization
    dist[i] = np.min(np.where(LRs >= 1/alpha)) # trials at which threshold is crossed

# report mean, median, and 90th percentile
print(np.mean(dist), np.percentile(dist, [50, 90]))

