from __future__ import division
import numpy as np
#from pyspark import SparkContext   # This is only needed for code which is run outside the Spark IPython notebook.

try:
    get_ipython().magic('matplotlib inline')
except:
    pass


# create some functions to run the regression:
def run_regression(X,Y):
    '''Simple function to execute linear regression'''
    # Set up matrix values for regression execution:
    XTX = np.dot(X.T, X)    # X'X
    XTY = np.dot(X.T, Y)    # X'Y

    # Linear-algebra solve for Beta in more numerically efficient way:
    Beta = np.linalg.solve(XTX, XTY)
    return Beta

def single_bootstrap(X,Y,seed=None):
    # Get Beta_hat
    Beta_hat = run_regression(X,Y)

    # Get eps_hat:
    eps_hat = Y - np.dot(X, Beta_hat)

    # Now do one bootstrap resampling:
    # Create random number generator:
    RNG = np.random.RandomState(seed)
    Nobs = eps_hat.shape[0]

    # Resample the eps_hat:
    epsilon_resample_index = RNG.randint(low=0,high=Nobs,size=Nobs)
    eps_hat_squiggle = np.array( [ eps_hat[epsilon_resample_index, 0] ] ).T

    # Create new Y_hat_squiggle:
    Y_hat_squiggle = np.dot(X, Beta_hat) + eps_hat_squiggle

    # Re-estimate Beta_hat using Y_hat_squiggle and X:
    Beta_hat_squiggle = run_regression(X,Y_hat_squiggle)

    return Beta_hat_squiggle

# Bring in the data:
import statsmodels.api as sm
from time import time
import pylab as plt
import numpy as np
import csv

# Load the data I want to use:
print "Load Geyser data exported from R's MASS public data libraries"
with open("./old_faithful_MASS_data.csv", 'rb') as filehandle:
    csv_reader = csv.reader(filehandle)
    header = csv_reader.next()   # Pull off the header
    old_faithful = []
    for row in csv_reader:
        old_faithful.append([float(x) for x in row])

geyser_data = zip(*old_faithful)
# Now have the data all together.

# Get number of observations:
N = len(geyser_data[0])

# Set up a column vector of y-values:
Y = np.array( [ geyser_data[header.index('duration')] ] ).T

# Set up matrix of X-values:
X = np.array( [ [1.0]*N, geyser_data[header.index('waiting')] ] ).T

print "Scatterplot of duration and waiting:"
plt.scatter(x=X[:,1], y=Y, s=20, c=u'b', marker=u'o')
plt.title("Waiting Time for Eruption vs.\nDuration of Subsequent Eruption")
plt.xlabel("Waiting time, minutes")
plt.ylabel("Duration, minutes")
plt.show()

print "\Histograms of duration and waiting:"
plt.hist(x=X[:,1], label="Waiting Times")
plt.title("Waiting Time for Eruption")
plt.xlabel("Minutes")
plt.show()

plt.hist(x=Y, label="Duration of Eruption")
plt.title("Duration of Eruption")
plt.xlabel("Minutes")
plt.show()

# --------------------------------------------------------------------------
# ------------ Run regression and bootstrap in single core mode ------------
# --------------------------------------------------------------------------

print "Running regression and bootstrap in single core mode, using previously defined 'run_regression' and 'single_bootstrap' functions."

N_boots = 100000
seed_seed = 123456

print "Step 1: Run the regression"
Beta_hat = run_regression(X,Y)

# Set up random number generation for bootstrap variance estimate
seed_RNG = np.random.RandomState(seed_seed)
parallel_seeds = seed_RNG.randint(0,2**30,size=N_boots)

print "Step 2: Run the bootstrap using map and previously-defined functions"
mapped_bootstrap = lambda seed, X=X,Y=Y:  single_bootstrap(X=X,Y=Y,seed=seed)
t0 = time()
results = map(mapped_bootstrap, parallel_seeds)
t1 = time()

# Pull out important values:
beta_std = np.array([ [np.std(z) for z in zip(*results)] ]).T

# Print results:
print "\n\nFinished funning single-core regression with bootstrap. Time and results as follows:"
single_core_bootstrap_time = (t1-t0)/60.0
print "\nTime: ", single_core_bootstrap_time, "min\nResults are: \nbeta:",Beta_hat, "\nbeta_std:", beta_std


# --------------------------------------------------------------------------
# --------------- Run basic OLS regression from Statsmodels  ---------------
# --------------------------------------------------------------------------

print "Now running regression using standard OLS methods and asymptotic variance as estimated by Python library statsmodels."

# Now run the same thing using statsmodels:
reg = sm.OLS(Y,X)
estimation = reg.fit()
print "Regression fit results from statsmodels:\n",estimation.summary2()

# *NOW* the Q: can we run this quickly across all cores using pyspark?
from pyspark import SparkContext

try:
    sc = SparkContext("local", appName="Basic-Bootstrap-OLS")   # Need to leave this out when
except:
    pass

# 
rdd_bootseeds = sc.parallelize([xseed for xseed in parallel_seeds])
# rdd_bootseeds = sc.parallelize([xseed for xseed in parallel_seeds])

# Define the massive function which will be sent across all the seeds:
# NOTE that have bound the data values.
mapped_bootstrap = lambda seed, X=X,Y=Y:  single_bootstrap(X=X,Y=Y,seed=seed)

trdd0 = time()
resultsRDD = rdd_bootseeds.map(mapped_bootstrap).collect()
trdd1 = time()

# .... and now look at the resultsRDD

# pull out the values:
#beta_std = np.array([ [np.std(z) for z in zip(*results)] ]).T

RDD_bootstrap_time = (trdd1-trdd0)/60.0
print "\nTime to Spark bootstrap: ", RDD_bootstrap_time, "min\n"

# pull out the values:
spark_beta_std = np.array([ [np.std(z) for z in zip(*resultsRDD)] ]).T

print "Results are: \nbeta:",Beta_hat, "\nsingle-core beta_std:", beta_std, "\nmulti-core beta_std:", spark_beta_std



