from scipy import stats

import scipy as sp
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

s = sp.randn(100) # Hundred random numbers from a standard Gaussian
print(len(s))

print("Mean : {0:8.6f}".format(s.mean()))
print("Minimum : {0:8.6f}".format(s.min()))
print("Maximum : {0:8.6f}".format(s.max()))
print("Variance : {0:8.6f}".format(s.var()))
print("Std. deviation : {0:8.6f}".format(s.std()))

print("Mean : {0:8.6f}".format(sp.mean(s)))
print("Variance : {0:8.6f}".format(sp.var(s)))
print("Std. deviation : {0:8.6f}".format(sp.std(s)))

print("Median : {0:8.6f}".format(sp.median(s)))

print("Variance with N in denominator = {0:8.6f}".format(sp.var(s)))
print("Variance with N - 1 in denominator = {0:8.6f}".format(sp.var(s,ddof=1)))

from scipy import stats

n, min_max, mean, var, skew, kurt = stats.describe(s)

print("Number of elements: {0:d}".format(n))
print("Minimum: {0:8.6f} Maximum: {1:8.6f}".format(min_max[0], min_max[1]))
print("Mean: {0:8.6f}".format(mean))
print("Variance: {0:8.6f}".format(var))
print("Skew : {0:8.6f}".format(skew))
print("Kurtosis: {0:8.6f}".format(kurt))

n = stats.norm(loc=3.5, scale=2.0)

n.rvs()

stats.norm.rvs(loc=3.5, scale=2.0)

# PDF of Gaussian of mean = 0.0 and std. deviation = 1.0 at 0.
stats.norm.pdf(0, loc=0.0, scale=1.0)

stats.norm.pdf([-0.1, 0.0, 0.1], loc=0.0, scale=1.0)

tries = range(11) # 0 to 10
print(stats.binom.pmf(tries, 10, 0.5))

def binom_pmf(n=4, p=0.5):
    # There are n+1 possible number of "successes": 0 to n.
    x = range(n+1)
    y = stats.binom.pmf(x, n, p)
    plt.plot(x,y,"o", color="black")

    # Format x-axis and y-axis.
    plt.axis([-(max(x)-min(x))*0.05, max(x)*1.05, -0.01, max(y)*1.10])
    plt.xticks(x)
    plt.title("Binomial distribution PMF for tries = {0} & p ={1}".format(
            n,p))
    plt.xlabel("Variate")
    plt.ylabel("Probability")

    plt.draw()

binom_pmf(n=10, p=0.5)

stats.norm.cdf(0.0, loc=0.0, scale=1.0)

def norm_cdf(mean=0.0, std=1.0):
    # 50 numbers between -3σ and 3σ
    x = sp.linspace(-3*std, 3*std, 50)
    # CDF at these values
    y = stats.norm.cdf(x, loc=mean, scale=std)

    plt.plot(x,y, color="black")
    plt.xlabel("Variate")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF for Gaussian of mean = {0} & std. deviation = {1}".format(
               mean, std))
    plt.draw()

norm_cdf()

stats.norm.ppf(0.5, loc=0.0, scale=1.0)

def norm_ppf(mean=0.0, std=1.0):
    # 100 numbers between 0 and 1.0 i.e., probabilities.
    x = sp.linspace(0, 1.0, 100)
    # PPF at these values
    y = stats.norm.ppf(x, loc=mean, scale=std)

    plt.plot(x,y, color="black")
    plt.xlabel("Cumulative Probability")
    plt.ylabel("Variate")

    plt.title("PPF for Gaussian of mean = {0} & std. deviation = {1}".format(
               mean, std))
    plt.draw()

norm_ppf()

stats.norm.sf(0.0, loc=0.0, scale=1.0)

def norm_sf(mean=0.0, std=1.0):
    # 50 numbers between -3σ and 3σ
    x = sp.linspace(-3*std, 3*std, 50)
    # SF at these values
    y = stats.norm.sf(x, loc=mean, scale=std)

    plt.plot(x,y, color="black")
    plt.xlabel("Variate")
    plt.ylabel("Probability")
    plt.title("SF for Gaussian of mean = {0} & std. deviation = {1}".format(
               mean, std))
    plt.draw()

norm_sf()

stats.norm.isf(0.5, loc=0.0, scale=1.0)

def norm_isf(mean=0.0, std=1.0):
    # 100 numbers between 0 and 1.0
    x = sp.linspace(0, 1.0, 100)
    # PPF at these values
    y = stats.norm.isf(x, loc=mean, scale=std)

    plt.plot(x,y, color="black")
    plt.xlabel("Probability")
    plt.ylabel("Variate")

    plt.title("ISF for Gaussian of mean = {0} & std. deviation = {1}".format(
               mean, std))
    plt.draw()

norm_isf()

# 100 random values from a Normal distribution with mu = 1.0
stats.norm.rvs(loc=0.0, scale=1.0, size=100)

# 100 random values from a Poisson distribution with mu = 1.0
stats.poisson.rvs(1.0, size=100)

import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

def simulate_poisson():
    # Mean is 1.69
    mu = 1.69
    sigma = sp.sqrt(mu)
    mu_plus_sigma = mu + sigma

    # Draw random samples from the Poisson distribution, to simulate
    # the observed events per 2 second interval.
    counts = stats.poisson.rvs(mu, size=100)

    # Bins for the histogram: only the last bin is closed on both
    # sides. We need one more bin than the maximum value of count, so
    # that the maximum value goes in its own bin instead of getting
    # added to the previous bin.
    # [0,1), [1, 2), ..., [max(counts), max(counts)+1]
    bins = range(0, max(counts)+2)

    # Plot histogram.
    plt.hist(counts, bins=bins, align="left", histtype="step", color="black")

    # Create Poisson distribution for given mu.
    x = range(0,10)
    prob = stats.poisson.pmf(x, mu)*100 

    # Plot the PMF.
    plt.plot(x, prob, "o", color="black")

    # Draw a smooth curve through the PMF.
    l = sp.linspace(0,11,100)
    s = interpolate.spline(x, prob, l)
    plt.plot(l,s,color="black")

    plt.xlabel("Number of counts per 2 seconds")
    plt.ylabel("Number of occurrences (Poisson)")

    # Interpolated probability at x = μ + σ; for marking σ in the graph.
    xx = sp.searchsorted(l,mu_plus_sigma) - 1
    v = ((s[xx+1] -  s[xx])/(l[xx+1]-l[xx])) * (mu_plus_sigma - l[xx])
    v += s[xx]

    ax = plt.gca()
    # Reset axis range and ticks.
    ax.axis([-0.5,10, 0, 40])
    ax.set_xticks(range(1,10), minor=True)
    ax.set_yticks(range(0,41,8))
    ax.set_yticks(range(4,41,8), minor=True)

    # Draw arrow and then place an opaque box with μ in it.
    ax.annotate("", xy=(mu,29), xycoords="data", xytext=(mu, 13),
                textcoords="data", arrowprops=dict(arrowstyle="->",
                                                   connectionstyle="arc3"))
    bbox_props = dict(boxstyle="round", fc="w", ec="w")
    ax.text(mu, 21, r"$\mu$", va="center", ha="center",
            size=15, bbox=bbox_props)

    # Draw arrow and then place an opaque box with σ in it.
    ax.annotate("", xy=(mu,v), xytext=(mu_plus_sigma,v),
                arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
    bbox_props = dict(boxstyle="round", fc="w", ec="w")
    ax.text(mu+(sigma/2.0), v, r"$\sigma$", va="center", ha="center",
            size=15, bbox=bbox_props)

    # Refresh plot and save figure.
    plt.draw()
    plt.savefig("simulate_poisson.png")

simulate_poisson()



