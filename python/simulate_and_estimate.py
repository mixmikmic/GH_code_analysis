# HIDDEN
from datascience import *
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plots
import numpy as np
plots.style.use('fivethirtyeight')
# datascience version number of last run of this notebook
version.__version__

# The magic number - size of the population that (in the real world) 
# we don't know and want to estimate

def createPopulation():
    def serNo(x):
        return "{:05d}".format(x)
    p = Table([np.arange(1,37*55)],["Ser No"])
    p.set_format("Ser No", serNo)
    return p

# Create a simulation of the population as a table - ordered collection of named columns
population = createPopulation()
population

# computational thinking - simulate observing a sample of the population
sample_size = 10

population.sample(sample_size,with_replacement=True)

# Simulate observing multiple samples
nsamples = 30

# use iteration to create a table of samples 
samples = Table()
for i in range(nsamples):
    name = "sample-"+str(i)
    a_sample = population.sample(sample_size,with_replacement=True)
    samples[name] = a_sample["Ser No"]
samples

# gracefully transition between tables and arrays
samples['sample-0']

# define a function to capture formally a idea about how to do the estimation
def estimateA(smpl) :
    return np.max(smpl)

estimateA(samples['sample-2'])

# you might come up with lots of other estimators
def estimateB(smpl) :
    return 2*np.mean(smpl)

#verify it works
estimateA(samples["sample-0"])

# illustrate list comprehension to explore data
[estimateB(samples[s]) for s in samples]

# Build a tables of estimates
estA = Table([[estimateA(samples[s]) for s in samples]],['ests'])
estA

# Look at the behavior of this estimator as a histogram
estA.hist(range=(1,np.max(estA['ests'])),bins=20)

# Computational thinking: estimator as a higher order function 
# passed in to a function that creates a table of estimate
def estimate(estimator):
    return Table([[estimator(samples[s]) for s in samples]],['ests'])

estB = estimate(estimateB)

estB.hist(range=(1,np.max(estB['ests'])),bins=20)

comp = Table([estA['ests'],estB['ests']],['estA','estB'])

comp

comp.hist(overlay=True, bins=np.arange(1000,2500,50))

# How does these estimates compare with the true size of the population?
population.num_rows

# Produce a table containing the data associated with a histogram
ebins = comp.bin(bins=np.arange(1000,2500,50))
ebins.show()



