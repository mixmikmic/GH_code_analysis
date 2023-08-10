sallieSVLs = [5.94,5.89,7.31,5.98,6.27,5.00,5.29,6.79,5.19,5.54,6.83,6.20,6.51,5.33,6.08]

from scipy.stats import norm

def like(data,mean,sd=0.7):   # Assuming standard deviation is known (0.7).
    """
    This function takes data as a list of continuous values and calculates the joint
    probability of those values assuming they are independent draws from a Normal
    distribution with the specified mean and standard deviation.
    """
    try:               # Note the use of the try...except pair to catch errors
        like = 1
        for d in data:
            like *= norm.pdf(d,loc=mean,scale=sd)
        return like
    except:
        print("You provided non-numeric data to the likelihood function. Try again...")
        raise

def hillClimb (data,param):
    """
    This hill-climbing function will find the maximum-likelihood estimate and score by
    relying on a function called like() to calculate likelihoods. It is only set up to 
    explore values of one parameter. The data can be defined in any way, as long as they
    are consistent with what the like() function expects.
    """
    
    diff = 0.1
    pCurr = param
    
    while ( diff > 0.001):
        L_pCurr = like(data,pCurr)
        L_pUp = like(data,pCurr+diff)
        L_pDown = like(data,pCurr-diff)

        while ( L_pUp > L_pCurr ):
            pCurr = pCurr + diff
            L_pCurr = like(data,pCurr)
            L_pUp = like(data,pCurr+diff)
            L_pDown = like(data,pCurr-diff)

        while (L_pDown > L_pCurr ):
            pCurr = pCurr - diff
            L_pCurr = like(data,pCurr)
            L_pUp = like(data,pCurr+diff)
            L_pDown = like(data,pCurr-diff)

        diff /= 2

    return (pCurr,L_pCurr)  # Returns tuple with BOTH MLE and likelihood score

svlML = hillClimb(sallieSVLs,5.6)
print("The MLE is %.2f cm with likelihood score %.4E." % svlML)

# Note the Python string formatting used above. Each of the elements in the string
# beginning with % indicates a placeholder for a value. These values are stored in the svlML
# tuple that follows after the string. A "%f" element indicates a float. A "%.2f" element
# indicates a float rounded to 2 decimal places. A "%E" element indicates a number with 
# scientific notation. Similarly, "%.4E" indicates scientific notation with 4 decimal places.

print("The likelihood score for our expected mean of 5.6 cm is: %.4E" % like(sallieSVLs,5.6))

print("The likelihood ratio is: %f" % (like(sallieSVLs,6.01)/like(sallieSVLs,5.6)) )

datasets = []

# Set the number of simulations. More simulations = greater precision, but more time.
numReps = 1000 

for _ in range(numReps):
    datasets.append(norm.rvs(loc=5.6,scale=0.7,size=15))

trueLikes = []

for rep in range(numReps):
    trueLikes.append(like(datasets[rep],mean=5.6))

maxLikes = []

for rep in range(numReps):
    if rep % 25 is 0:
        print(" %s " % rep,end="")
    MLvals = hillClimb(datasets[rep],5.6)
    maxLikes.append(MLvals[1])

LRs = []

for rep in range(numReps):
    LRs.append(maxLikes[rep]/trueLikes[rep])

import matplotlib.pyplot as plt

numBins = 1000

n, bins, patches = plt.hist(LRs, numBins, normed=1, facecolor='blue', alpha=0.75)
plt.xlabel('Likelihood Ratio')
plt.ylabel('Frequency')
plt.axis([1, 15, 0, max(n)+0.1])
plt.grid(True)

# A vertical line will indicate the position of our observed LR
obsLR = (like(sallieSVLs,6.01)/like(sallieSVLs,5.6))
plt.plot([obsLR,obsLR], [0, 2], 'r-', lw=5)

plt.show()

# Empirical likelihood ratio shown in red

extCount = 0        # Initializing a counter for the proportion of extreme values
for LR in LRs:
    if LR > obsLR:
        extCount += 1  # Incrementing counter when simulated value greater than observed
extCount/numReps    # Printing the proportion that are greater than our observed.

import math  # Importing math library to calculate logs

LRs_sorted = sorted(LRs) # Sorting our likelihood ratios

# Finding the value in the 95th percentile of our sorted list
critValue = LRs_sorted[round(0.95*len(LRs_sorted))]

print("alpha_0.05 = %s" % critValue)

2*math.log(critValue)

LRs_log = [2*math.log(LR) for LR in LRs]

numBins = 50

n, bins, patches = plt.hist(LRs_log, numBins, normed=1, facecolor='blue', alpha=0.75)
plt.xlabel('Log-Likelihood Ratio')
plt.ylabel('Frequency')
plt.axis([0, 10, 0, max(n)+0.1])
plt.grid(True)

# A vertical line will indicate the position of our observed LR
plt.plot([2*math.log(obsLR),2*math.log(obsLR)], [0, 2], 'r-', lw=5)

plt.show()

# Empirical likelihood ratio shown in red

