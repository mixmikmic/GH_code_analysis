import numpy as np
import matplotlib.pyplot as plt


# probability distribution we're trying to calculate
p = lambda x: np.exp(-x)*(x>0)

# CDF of p or exp_cdf(arr,1.)
CDF = lambda x: (1-np.exp(-x))*(x>0)

# invert the CDF
invCDF = lambda u: -np.log(1-u)

 # the total of samples we wish to generate
N = 10000

# generate uniform samples in our range then invert the CDF
# to get samples of our target distribution
R = np.random.rand(N)#np.random.uniform(rmin, rmax, N)
X = invCDF(R)


# plot the histogram
plt.figure(figsize=(8,5))

plt.hist(X, bins=50,  normed=1, label=u'Inverse Sampling Hist', 
         edgecolor='black', linewidth=1.2);

# plot our (normalized) function
xvals=np.linspace(1e-5, 5, 1000)

plt.plot(xvals, p(xvals), 'b', label=u'actual probability density exp(-x)')
plt.style.use('ggplot')
# turn on the legend
plt.legend()
plt.show()

# We consider the following array of values a --200 of them
aa = np.linspace(1e-5, 5, 200)

#for each number A in a, 
# we want to calculate the number of samples < A / total number of samples
# store the result in CDFarr

CDFarr = np.zeros(len(aa))

for i in range(0,len(aa)):
    A = aa[i]
    total     = np.sum(X<A)
    prob_A    = total/len(X)
    CDFarr[i] = prob_A
    


plt.figure(figsize=(10,10))

plt.subplot(211)
plt.plot(aa,CDFarr,'.',color = 'blue',
         label = u'probability calculation using sample portion')
plt.legend()

plt.subplot(212)
plt.plot(aa,CDF(aa),'-',
        label = u'actual probability distribution 1-e^(-x)')
plt.legend()

plt.show()



