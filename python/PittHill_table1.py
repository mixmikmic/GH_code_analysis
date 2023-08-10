# some programmatic housekeeping
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
np.random.seed(215)
get_ipython().magic('matplotlib inline')

notebook = "PittHill_table1.ipynb" # replace with FILENAME
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(notebook), '..', 'data')))

def PoissonTriples_exact(moment, N):
    """
    This function computes the probability that triples of Poisson random variables
    contain their own rounded mean based on the formula given in Pitt & Hill, 2016.
    
    Parameters
    ----------
    moment : integer
              The mean-variance parameter of the Poisson distribution from which
              triples of Poisson random variables are to be generated.
         
    N : integer
         The...
    
    Returns
    -------
    prob : numeric
            The exact probability that triples of Poisson random variables contain
            their own rounded means.
    """
    for j in list(range(2, N + 1)):
        for k in list(range(j, N + 1)):
            inner = poisson.pmf(k - (j / 2), moment) + ((j % 2) * poisson.pmf(k - (j / 2) - 1, moment))
            outer = poisson.pmf(k, moment) * poisson.pmf(k - j, moment)
            prob = outer * inner
    
    prob = 6 * prob
    return(prob)

def PoissonTriples_empirical(moment, n_times):
    """
    This function computes the probability that triples of Poisson random variables
    contain their own rounded mean.
    
    Parameters
    ----------
    moment : integer
              The mean-variance parameter of the Poisson distribution from which
              triples of random variables are to be generated.
    n_times : integer
               The number of Poisson triples (from a distribution parameterized by
               the moment argument) to use in computing the probability.
               
    Returns
    -------
    prob : numeric
            The empirically computed probability that triples of Poisson random
            variables contain their rounded own means.
    """
    inCounter = 0
    
    for i in range(n_times):
        poisTriple = poisson.rvs(moment, size = 3)
        tripleMean = int(np.round(np.mean(poisTriple), decimals = 0))
        meanCheck = np.sum(np.in1d(poisTriple, tripleMean))
    
        if meanCheck != 0:
            inCounter += 1
    
    prob = inCounter / n_times
    return(prob)

nTriples = 1000 # number of RV triples to generate to compute the empirical probability
poisMoments = range(1, 1000) # each mean-variance parameter of the Poisson distribution
probs = np.zeros(len(poisMoments))

# Computing the inclusion probability of the mean using (nTriples) of Poisson RV triples
# across as many values of the mean-variance parameter as specified above. This loop takes
# a while to run -- on the order of 2-5 minutes...
for i in range(len(poisMoments)):
    probs[i] = PoissonTriples_empirical(poisMoments[i], nTriples)

plt.figure(num = None, figsize=(12, 9), dpi=80)
plt.plot(poisMoments, probs)
plt.xlabel('$\lambda$ (moment) parameter of Poisson distribution')
plt.ylabel('Probability')
plt.title(r'Histogram of Inclusion Probabilities from Poisson Triples')
plt.axis([min(poisMoments), max(poisMoments), 0, min(1.5 * max(probs), 1)])
plt.show()

probs_df = pd.DataFrame(data = [poisMoments, probs]).transpose()
probs_df.columns = ["$\lambda$", "Inclusion Probability"]
table1 = probs_df.head(25)
table1

