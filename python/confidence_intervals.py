import math
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

VERBOSE = True

def phi(x,mu=0,sigma=1):
    """ Cumulative distribution function for normal distribution """
    return (1.0 + math.erf((x - mu)/ (sigma*math.sqrt(2.0)))) / 2.0

mu = 500
std = 15
sample_size=100
alpha = 0.05

def confidence_interval(mu,std,sample_size,critical_val=1.96):
    """Compute a confidence interval for a normal distribution. If population sigma (standard deviation) is known,
    then the critical_val should be the z_score. Otherwise, critical_val should use Student's t distribution to 
    calculate the interval. (t is z as sample size approaches infinity; when sample_size >= 30, people often use Z
    rather than t) """
    interval = critical_val*(std/math.sqrt(sample_size))
    if VERBOSE:
        print(round(mu - critical_val*(std/math.sqrt(sample_size)),2),round(mu + critical_val*(std/math.sqrt(sample_size)),2))
    return mu-interval,mu+interval
        

confidence_interval(mu=mu,std=std,sample_size=sample_size,critical_val=1.96)

stats.norm.interval(alpha, loc=mu, scale=std/math.sqrt(100))

sample = np.random.normal(loc=mu, scale=std, size=sample_size)
if VERBOSE: print (np.mean(sample),np.std(sample))

repeats = 100 #not sure how to choose the number of bootstrapping steps

means = []
stds = []
for r in range(repeats):
    boot = np.random.choice(sample,sample_size,replace=True) #bootstrapping draws with replacement
    means.append(np.mean(boot))
    stds.append(np.std(boot))
if VERBOSE: print(np.mean(means),np.mean(stds))

#to compute the confidence interval, one simply takes the empirical quantiles from the bootstrap distribution of the parameter
confi = np.percentile(means,[(alpha/2)*100,(1-(alpha/2))*100])
if VERBOSE: print (confi)

#This is essentially the logical behind percentile, although the built-in function uses interpolation
#again, this is just for learning purposes
means = sorted(means)
start = int(math.floor((alpha/2)*sample_size))
finish = int(math.ceil((1-(alpha/2))*sample_size))
if VERBOSE: print (means[start],means[finish])

def plot_confidence_interval_histogram(dist, myrange,bins=30):
    fig,ax = plt.subplots()
    ax.hist(dist,bins)
    #ax.axhline(y=1,xmin=myrange[0],xmax=myrange[1],color='r',linewidth=4)
    plt.plot([myrange[0], myrange[1]], [1, 1],color='r',linewidth=2)
    plt.show()
    
plot_confidence_interval_histogram(means,confi)
plt.title('bootstrapped mean values')

plot_confidence_interval_histogram(sample,confi)
plt.title('sample distribution with 95% confidence interval around mean')

stats.bayes_mvs(sample, alpha=alpha)

