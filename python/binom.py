# This is the first cell with code: set up the Python environment
get_ipython().run_line_magic('matplotlib', 'inline')
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

def binoLowerCL(n, x, cl = 0.975, inc=0.000001, p = None):
    "Lower confidence level cl confidence interval for Binomial p, for x successes in n trials"
    if p is None:
            p = float(x)/float(n)
    lo = 0.0
    if (x > 0):
            f = lambda q: cl - scipy.stats.binom.cdf(x-1, n, q)
            lo = sp.optimize.brentq(f, 0.0, p, xtol=inc)
    return lo

def binoUpperCL(n, x, cl = 0.975, inc=0.000001, p = None):
    "Upper confidence level cl confidence interval for Binomial p, for x successes in n trials"
    if p is None:
            p = float(x)/float(n)
    hi = 1.0
    if (x < n):
            f = lambda q: scipy.stats.binom.cdf(x, n, q) - (1-cl)
            hi = sp.optimize.brentq(f, p, 1.0, xtol=inc) 
    return hi

# Population of two values, {0, 1}, in various proportions.  Amounts to Binomial random variable
ns = np.array([25, 50, 100, 400])  # sample sizes
ps = np.array([.001, .01, 0.1])    # mixture fractions, proportion of 1s in the population
alpha = 0.05  # 1- (confidence level)
reps = int(1.0e3)   # just for demonstration
vals = [0, 1]

simTable = pd.DataFrame(columns=('fraction of 1s', 'sample size', 'Student-t cov',                                 'Binom cov', 'Student-t len', 'Binom len'))
for p in ps:
    popMean = p
    for n in ns:
        tCrit = sp.stats.t.ppf(q=1.0-alpha/2, df=n-1)
        samMean = np.zeros(reps)
        sam = sp.stats.binom.rvs(n, p, size=reps)
        samMean = sam/float(n)
        samSD = np.sqrt(samMean*(1-samMean)/(n-1))
        coverT = (np.fabs(samMean-popMean) < tCrit*samSD).sum()
        aveLenT = 2*(tCrit*samSD).mean()
        coverB = 0
        totLenB = 0.0
        for r in range(int(reps)):  
            lo = binoLowerCL(n, sam[r], cl=1.0-alpha/2)
            hi = binoUpperCL(n, sam[r], cl=1.0-alpha/2)
            coverB += ( p >= lo) & (p <= hi)
            totLenB += hi-lo
        simTable.loc[len(simTable)] =  p, n, str(100*float(coverT)/float(reps)) + '%',             str(100*float(coverB)/float(reps)) + '%',            str(round(aveLenT,4)),            str(round(totLenB/float(reps),4))
#
ansStr =  '<h3>Simulated coverage probability and expected length of Student-t and Binomial confidence intervals for a {0, 1} population</h3>' +          '<strong>Nominal coverage probability ' + str(100*(1-alpha)) +          '%</strong>.<br /><strong>Estimated from ' + str(int(reps)) + ' replications.</strong>'
display(HTML(ansStr))
display(simTable)

# Nonstandard mixture: a pointmass at zero and a uniform[0,1]
ns = np.array([25, 50, 100, 400])  # sample sizes
ps = np.array([0.9, 0.99, 0.999])    # mixture fraction, weight of pointmass
thresh = [0.2, 0.1, 0.01, .001]
alpha = 0.05  # 1- (confidence level)
reps = 1.0e3   # just for demonstration

cols = ['mass at 0', 'sample size', 'Student-t cov']
for i in range(len(thresh)):
    cols.append('Bin t=' + str(thresh[i]) + ' cov')
cols.append('Student-t len')
for i in range(len(thresh)):
    cols.append('Bin t=' + str(thresh[i]) + ' len')


simTable = pd.DataFrame(columns=cols)

for p in ps:
    popMean = (1-p)*0.5  #  p*0 + (1-p)*.5
    for n in ns:
        tCrit = sp.stats.t.ppf(q=1-alpha, df=n-1)
        coverT = 0    # coverage of t intervals
        tUp = 0       # mean upper bound of t intervals
        coverB = np.zeros(len(thresh))  # coverage of binomial threshold intervals
        bUp = np.zeros(len(thresh))     # mean upper bound of binomial threshold intervals
        for rep in range(int(reps)):
            sam = np.random.uniform(size=n)
            ptMass = np.random.uniform(size=n)
            sam[ptMass < p] = 0.0
            samMean = np.mean(sam)
            samSD = np.std(sam, ddof=1)
            tlim = samMean + tCrit*samSD
            coverT += (popMean <= tlim)  # one-sided Student-t
            tUp += tlim
            for i in range(len(thresh)):
                x = (sam > thresh[i]).sum()  # number of binomial "successes"
                pPlus = binoUpperCL(n, x, cl=1-alpha)
                blim = thresh[i]*(1.0-pPlus) + pPlus
                coverB[i] += (popMean <= blim)
                bUp[i] += blim
        theRow = [p, n, str(100*float(coverT)/float(reps)) + '%']
        for i in range(len(thresh)):
            theRow.append(str(100*float(coverB[i])/float(reps)) + '%')
        theRow.append(str(round(tUp/float(reps), 3)))
        for i in range(len(thresh)):
            theRow.append(str(round(bUp[i]/float(reps), 3)))
        simTable.loc[len(simTable)] =  theRow
#
ansStr =  '<h3>Simulated coverage probability and expected lengths of one-sided Student-t confidence intervals and threshold ' +          'Binomial intervals for mixture of U[0,1] and pointmass at 0</h3>' +          '<strong>Nominal coverage probability ' + str(100*(1-alpha)) +          '%</strong>. <br /><strong>Estimated from ' + str(int(reps)) + ' replications.</strong>'

display(HTML(ansStr))
display(simTable)

get_ipython().run_line_magic('run', 'talkTools.py')



