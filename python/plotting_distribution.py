fname = 'terrorism.txt'

get_ipython().system("head 'terrorism.txt'")

get_ipython().system("tail 'terrorism.txt'")

nums = [int(x) for x in open(fname)]
print(nums[:5], '...', nums[-5:])

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from collections import Counter

a = Counter([1,1,1,1,2,2])
a

N = len(nums)
N

nk = Counter(nums)

print(nk[1], nk[2])

x = []
y = []
for k in sorted(nk):
    x.append(k)
    y.append(nk[k])

plt.scatter(x,y)

plt.plot(x,y)

plt.ylim((0.7, 10000)) # more clearly show the points at the bottom. 
plt.xlabel('Number of deaths, x')
plt.ylabel("n(X)")
plt.loglog(x,y, 'o', markersize=3, markerfacecolor='none')

import numpy as np

bins = np.logspace(0.0, 4.0, num=40)
bins

ax = plt.subplot(1,1,1)

ax.set_xlabel('Number of deaths, x')
ax.set_ylabel('p(x)')
ax.hist(nums, bins=bins, normed=True)
ax.set_xscale('log')
ax.set_yscale('log')

Y, X = np.histogram(nums, bins=bins, normed=True)

X = [x*np.sqrt(bins[1]) for x in X][:-1]  # find the center point for each bin. can you explain this?

plt.ylim((0.00001, 1))
plt.xlabel('Number of deaths, x')
plt.ylabel("n(X)")
plt.loglog(X,Y, 'o', markersize=3, markerfacecolor='none')

# your code and results



