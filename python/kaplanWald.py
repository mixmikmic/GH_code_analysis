# This is the first cell with code: set up the Python environment
get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function, division
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy as sp
import scipy.stats
from scipy.stats import binom
import scipy.optimize
import pandas as pd
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import clear_output, display, HTML

def kaplanWaldLowerCI(x, cl = 0.95, gamma = 0.99, xtol=1.e-12, logf=True):
    '''
       Calculates the Kaplan-Wald lower 1-alpha confidence bound for the mean of a nonnegative random
       variable.
    '''
    alpha = 1.0-cl
    if any(x < 0):
        raise ValueError('Data x must be nonnegative.')
    elif all(x <= 0):
        lo = 0.0
    else:
        if logf:
            f = lambda t: (np.max(np.cumsum(np.log(gamma*x/t + 1 - gamma))) + np.log(alpha))
        else:
            f = lambda t: (np.max(np.cumprod(gamma*x/t + 1 - gamma)) - 1/alpha)
        xm = np.mean(x)
        if f(xtol)*f(xm) > 0.0:
            lo = 0.0
        else:
            lo = sp.optimize.brentq(f, xtol, np.mean(x), xtol=xtol) 
    return lo

# Nonstandard mixture: a pointmass at 1 and a uniform[0,1]
ns = np.array([25, 50, 100, 400])  # sample sizes
ps = np.array([0.9, 0.99, 0.999])    # mixture fraction, weight of pointmass
alpha = 0.05  # 1- (confidence level)
reps = int(1.0e4)   # just for demonstration
gamma = 0.99
xtol = 1.e-12

simTable = pd.DataFrame(columns=('mass at 1', 'sample size', 'Student-t cov',                                 'Kaplan-Wald cov', 'Student-t low', 'Kaplan-Wald low')
                       )

for p in ps:
    popMean = (1-p)*0.5 + p  # population is a mixture of uniform with mass (1-p) and
                             # pointmass at 1 with mass p
    
    for n in ns:
        tCrit = sp.stats.t.ppf(q=1-alpha, df=n-1)
        coverT = 0
        coverK = 0
        lowT = 0.0
        lowK = 0.0
        
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
            #  Kaplan-Wald interval
            kLo = kaplanWaldLowerCI(sam, cl=1-alpha, gamma=gamma, xtol=xtol) # lower endpoint of the Kaplan-Wald interval
            coverK += (kLo <= popMean )
            lowK += kLo

        simTable.loc[len(simTable)] =  p, n,            str(100*float(coverT)/float(reps)) + '%',            str(100*float(coverK)/float(reps)) + '%',            str(round(lowT/float(reps),4)),            str(round(lowK/float(reps),4))
#
ansStr =  '<h3>Simulated coverage probability and expected lower endpoint of ' +          'one-sided Student-t and Kaplan-Wald confidence intervals for ' +          'mixture of U[0,1] and pointmass at 1 population</h3>' +          '<strong>Nominal coverage probability ' + str(100*(1-alpha)) +          '%</strong>. Gamma=' + str(gamma) +          '.<br /><strong>Estimated from ' + str(int(reps)) +          ' replications.</strong>'

display(HTML(ansStr))
display(simTable)

