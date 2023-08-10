from scipy.stats import bernoulli, poisson, binom, norm, mvn
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

headimg = plt.imread('../data/quarterheads.jpg')
tailimg = plt.imread('../data/quartertails.jpg')

p = 0.1
# let us draw a sample from a bernoulli distribution
b = bernoulli.rvs(p,size=1)
print b
if b[0] == 0:
    plt.imshow(tailimg)
    plt.axis('off')
else:
    plt.imshow(headimg)
    plt.axis('off')
    

# you can also draw samples simultaneously
samples = bernoulli.rvs(p,size=1000)
print samples
# count the number of successes (sample = 1). What happens when you change p?
print np.count_nonzero(samples==1)

# plotting the probability mass function for the Bernoulli distribution
a = np.arange(2)

colors = matplotlib.rcParams['axes.color_cycle']
plt.figure(figsize=(12,8))
for i, p in enumerate([0.1, 0.2, 0.6, 0.7]):
    ax = plt.subplot(1, 4, i+1)
    plt.bar(a, bernoulli.pmf(a, p), label=p, color=colors[i], alpha=0.2)
    ax.xaxis.set_ticks(a)

    plt.legend(loc=0)
    if i == 0:
        plt.ylabel("PDF at $k$")
    

plt.suptitle("Bernoulli probability")

#sampling from a binomial distribution
sample = binom.rvs(20,0.4,1)
print sample

#plotting the pmf for a bernoulli distribution
plt.figure(figsize=(12,6))
k = np.arange(0, 22)
for p, color in zip([0.1, 0.3, 0.6, 0.8], colors):
    rv = binom(20, p)
    plt.plot(k, rv.pmf(k), lw=2, color=color, label=p)
    plt.fill_between(k, rv.pmf(k), color=color, alpha=0.3)
plt.legend()
plt.title("Binomial distribution")
plt.tight_layout()
plt.ylabel("PDF at $k$")
plt.xlabel("$k$")

# generate samples from a multinoulli distribution. Essentially simulated a single roll of dice. Note that the output is a vector of length $k = 6$
np.random.multinomial(1, [1/6.]*6, size=1)

# generate samples from a multinomial distribution. Note that the output is a vector of length $k = 6$
np.random.multinomial(20, [1/6.]*6, size=1)

# set the parameters
mu = 0
sigma = 1
# draw 1000 samples from this distribution
samples = norm(mu, sigma).rvs(1000)
# plot an empirical distribution, i.e., a histogram
hist(samples, 30, normed=True, alpha=.3)

# Compute the density at several instances of the random variable
x = linspace(-4, 4, 10001)
# plot the density
plot(x, norm(0, 1).pdf(x), linewidth=2)

#define the parameters for D = 2
mu = np.array([0,0])
Sigma = np.array([[1,0],[0,1]])
rv = multivariate_normal(mu,Sigma)
#sample some points
s = np.random.multivariate_normal(mu,Sigma,100)

fig = plt.figure()
plt.subplot(111)
plt.scatter(s[:,0],s[:,1])

# add a contour plot
smin = np.min(s,axis=0)
smax = np.max(s,axis=0)
t1=linspace(smin[0],smax[0],1000)
t2=linspace(smin[1],smax[1],1000)

# evaluate pdf at each of these mesh points

