# This is the first cell with code: set up the Python environment
get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import division
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy as sp
import scipy.stats
from scipy.stats import binom
import pandas as pd
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import clear_output, display, HTML

def ecdf(x):
    '''
       calculates the empirical cdf of data x
       returns the unique values of x in ascending order and the cumulative probabity at those values
       NOTE: This is not an efficient algorithm: it is O(n^2), where n is the length of x. 
       A better algorithm would rely on the Collections package or something similar and could work
       in O(n log n)
    '''
    theVals = sorted(np.unique(x))
    theProbs = np.array([sum(x <= v) for v in theVals])/float(len(x))
    if (theVals[0] > 0.0):
        theVals = np.append(0., theVals)
        theProbs = np.append(0., theProbs)
    return theVals, theProbs


def plotUniformKolmogorov(n):
    sam = np.random.uniform(size=n)
    v, pr = ecdf(sam)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot([0, 1], [0, 1], linestyle='--', color='r', label='true cdf')
    ax.step(v, pr, color='b', where='post', label='ecdf')
    ax.legend(loc='best')
    dLoc = np.argmax(v-pr)
    dMinus = (v-pr)[dLoc]
    ax.axvline(x=v[dLoc], ymin=pr[dLoc], ymax=v[dLoc], color='g', linewidth='4')
    ax.text(0.5, 0.1, r'$D_n^-=$' + str(round(dMinus, 3)), color='g', weight='heavy')

interact(plotUniformKolmogorov, n=widgets.IntSlider(min=3, max=300, step=1, value=10))
    

# find lower confidence bound on the mean of a non-negative population using the MDKW inequality

def massartOneSide(n, alpha):
    ''' 
       tolerable misfit between the cdf and ecdf for a one-sided bound 
       at significance level alpha for an iid sample of size n
    '''
    return np.sqrt(-np.log(alpha)/(2.0*n))
    
    
def ksLowerMean(x, c):
    '''
       lower confidence bound for the mean of a nonnegative population
       x is an iid sample with replacement from the population
       c is the Massart constant for the desired coverage
    '''
    # find the ecdf
    vals, probs = ecdf(x)
    probs = np.fmin(probs+c, 1)   # This is G^-
    gProbs = np.diff(np.append([0.0], probs))  # pre-pend a 0 so that diff does the right thing; 
                                               # gProbs is the vector of masses
    return (vals*gProbs).sum()

# Nonstandard mixture: a pointmass at 1 and a uniform[0,1]
ns = np.array([25, 50, 100, 400])  # sample sizes
ps = np.array([0.9, 0.99, 0.999])    # mixture fraction, weight of pointmass
alpha = 0.05  # 1- (confidence level)
reps = int(1.0e4) # just for demonstration

simTable = pd.DataFrame(columns=('mass at 1', 'sample size', 'Student-t cov',                                 'MDKW cov', 'Student-t low', 'MDKW low')
                       )

for p in ps:
    popMean = (1-p)*0.5 + p  # population is a mixture of uniform with mass (1-p) and
                             # pointmass at 1 with mass p
    
    for n in ns:
        tCrit = sp.stats.t.ppf(q=1-alpha, df=n-1)
        mCrit = np.sqrt(-np.log(alpha)/(2.0*n))  # the 1-sided MDKW constant
        coverT = 0
        coverM = 0
        lowT = 0.0
        lowM = 0.0
        
        for rep in range(int(reps)):
            sam = np.random.uniform(size=n)  # the uniform part of the sample
            ptMass = np.random.uniform(size=n)  # for a bunch of p-coin tosses
            sam[ptMass < p] = 1.0   # mix in pointmass at 1, with probability p
            # t interval
            samMean = np.mean(sam)
            samSD = np.std(sam, ddof=1)
            tLo = samMean - tCrit*samSD  # lower endpoint of the t interval
            coverT += ( tLo <= popMean )
            lowT += tLo
            #  MDKW interval
            mLo = ksLowerMean(sam, mCrit) # lower endpoint of the MDKW interval
            coverM += (mLo <= popMean )
            lowM += mLo

        simTable.loc[len(simTable)] =  p, n,            str(100*float(coverT)/float(reps)) + '%',            str(100*float(coverM)/float(reps)) + '%',            str(round(lowT/float(reps),4)),            str(round(lowM/float(reps),4))
#
ansStr =  '<h3>Simulated coverage probability and expected lower endpoint of ' +          'one-sided Student-t and MDKW confidence intervals for ' +          'mixture of U[0,1] and pointmass at 1 population</h3>' +          '<strong>Nominal coverage probability ' + str(100*(1-alpha)) +          '%</strong>.<br /><strong>Estimated from ' + str(int(reps)) + ' replications.</strong>'

display(HTML(ansStr))
display(simTable)

get_ipython().run_line_magic('run', 'talkTools.py')



