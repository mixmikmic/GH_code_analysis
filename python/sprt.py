# This is the first cell with code: set up the Python environment
get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function, division
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy as sp
import scipy.stats
from scipy.stats import binom
import pandas as pd
# For interactive widgets
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import clear_output, display, HTML

def plotBinomialSPRT(n, p, p0, p1, alpha, beta):
    '''
       Plots the progress of the SPRT for n iid Bernoulli trials with probabiity p
       of success, for testing the hypothesis that p=p0 against the hypothesis p=p1
       with significance level alpha and power 1-beta
    '''
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    trials = sp.stats.binom.rvs(1, p, size=n+1)  # leave room to start at 1
    terms = np.ones(n+1)
    sfac = p1/p0
    ffac = (1.0-p1)/(1.0-p0)
    terms[trials == 1.0] = sfac
    terms[trials == 0.0] = ffac
    terms[0] = 1.0
    logterms = np.log(terms)
    #
    ax[0].plot(range(n+1),np.cumprod(terms), color='b')
    ax[0].axhline(y=(1-beta)/alpha, xmin=0, xmax=n, color='g', label=r'$(1-\beta)/\alpha$')
    ax[0].axhline(y=beta/(1-alpha), xmin=0, xmax=n, color='r', label=r'$\beta/(1-\alpha)$')
    ax[0].set_title('Simulation of Wald\'s SPRT')
    ax[0].set_ylabel('LR')
    ax[0].legend(loc='best')
    #
    ax[1].plot(range(n+1),np.cumsum(logterms), color='b', linestyle='--')
    ax[1].axhline(y=math.log((1-beta)/alpha), xmin=0, xmax=n, color='g', label=r'$log((1-\beta)/\alpha)$')
    ax[1].axhline(y=math.log(beta/(1-alpha)), xmin=0, xmax=n, color='r', label=r'$log(\beta/(1-\alpha))$')
    ax[1].set_ylabel('log(LR)')
    ax[1].set_xlabel('trials')
    ax[1].legend(loc='best')
    plt.show()


interact(plotBinomialSPRT, n=widgets.IntSlider(min=5, max=300, step=5, value=100),         p=widgets.FloatSlider(min=0.001, max=1, step=0.01, value=.45),         p0=widgets.FloatSlider(min=0.001, max=1, step=0.01, value=.5),         p1=widgets.FloatSlider(min=0.001, max=1, step=0.01, value=.6),         alpha=widgets.FloatSlider(min=0.001, max=1, step=0.01, value=.05),         beta=widgets.FloatSlider(min=0.001, max=1, step=0.01, value=.05)
         )



