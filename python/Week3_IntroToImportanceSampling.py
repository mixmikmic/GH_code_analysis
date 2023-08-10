# Importing the binomial and uniform classes from the stats module of the scipy library
from scipy.stats import binom, uniform

# Importing pseudo-random number generators for uniform and Gaussian distributions
from random import random, gauss

# Defining the data
flips = ["H","T","T","T","H"]
# flips = flips*100  # Uncomment (and modify) this line to create a more informative dataset
n = len(flips)        # Number (n) of binomial trials
k = sum([1 for fl in flips if fl == "H"])  # Counting heads as successes (k)

# Defining general function to calculate likelihoods if p<0 or p>1
def like(successes,trials,prob,testingPrior=False):
    if testingPrior:  # If True, this will always return 1. This can be useful if one wants
        return 1      # to test the machinery by estimating a known distribution (the prior).
    if prob < 0:
        return 0
    elif prob > 1:
        return 0
    else:
        return binom.pmf(successes,trials,prob)
    
# Defining function to calculate prior density - uniform [0,1]
def prior(prob):
    return uniform.pdf(prob)

# Defining function to calculate the unnormalized posterior density
def posterior(successes,trials,prob):
    posterior = prior(prob) * like(successes,trials,prob)
    return posterior

# To get a sense for how well importance sampling is working, we're going to run our
# experiment several times. This list will hold the estimates of the means for all 
# replicates.
estimates = []

# The number of replicates we will run.
numReps = 100

# This value establishes the upper end of the uniform distribution from which we will sample
# parameter values. Since only values between 0 and 1 can have likelihoods > 0, the more we
# extend this value above 1, the greater the disparity between our sampling distribution and
# distribution of interest. What happens as this gets bigger?
uniScale = 1

# Initializing our ad hoc progress bar
print("Progress (each . = 10 replicates): ",end="")

# Iterating across our replicates 0,...,numReps-1
for rep in range(numReps):
    
    # Incrementing our progress
    if rep % 10 == 0:
        print(".",end="")
    
    # Draw values from uniform prior using the uniform class we imported from scipy
    numValues = 100
    p_vals = uniform.rvs(size=numValues,loc=0,scale=uniScale)
    
    # Calculate initial weights (not necessarily normalized)
    weights = [(posterior(k,n,param)/uniform.pdf(param,loc=0,scale=uniScale)) for param in p_vals]
    
    # Normalize weights so average is 1
    """NOTE: This normalization isn't strictly necessary if both functions used to calculate 
           the initial weights are proper probability density functions. But even then, it 
           helps with rounding error."""
    weights = [w/np.mean(weights) for w in weights]
        
    # Calculating weighted average
    count = 0
    estMean = 0
    while (count < (len(p_vals))):
        estMean += p_vals[count]*weights[count]  # Multiplying each value by its weight
        count += 1
    estMean /= numValues
    estimates.append(estMean)
    
# Printing out some useful summary information
print()
print("The last estimated mean (replicate %d): %f." % (numReps,estimates[numReps-1]))
print("The mean of the estimated means across all %d replicates: %f." % (numReps,np.mean(estimates)))
print("The standard deviation of the estimated means is: %f."% (np.std(estimates)))


# Rerun this several times, varying the size of the dataset, the upper end of the sampling
# distribution, and the number of values drawn (numValues) from the sampling distribution. 
# Pay attention to whether/how several things change:
# - Is an error thrown?
# - The estimated means
# - The standard deviation of the estimated means
# - The run time

# Now try changing the testingPrior argument to True for the likelihood function.
# What happens? Why might this be useful?

