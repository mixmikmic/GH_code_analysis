import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
plt.style.use('notebook');
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
colors = ["#2078B5", "#FF7F0F", "#2CA12C", "#D72827", "#9467BE", "#8C574B",
            "#E478C2", "#808080", "#BCBE20", "#17BED0", "#AEC8E9", "#FFBC79", 
            "#98E08B", "#FF9896", "#C6B1D6", "#C59D94", "#F8B7D3", "#C8C8C8", 
           "#DCDC8E", "#9EDAE6"]

def lcg_rand(a,c,m,seed,N=1):
    '''A linear congruential pseudrandom number generator'''
    x = np.zeros([N])
    X = seed
    x[0] = X/m
    for n in range(N-1):
        X = (a*X + c) % m
        x[n+1] = X/m
    return x

N = 1000
a,c,m,seed = 16807,0,2**31-1,332
m = 2**32-1
a = 7
c = 7
seed = 7
x = lcg_rand(a,c,m,seed,N)
print(x)

# the histogram of the data
n, bins, patches = plt.hist(x, 20, normed=1, ec='w')
plt.xlabel('x')
plt.ylabel('p(x)')

a,c,m,seed = 1664525,1013904223,2**32,13523
x = lcg_rand(a,c,m,seed,10000)

print(np.average(x),np.std(x))

plt.figure(figsize=(5,5))
plt.plot(x[:-1],x[1:],'o', ms=3, mew=0)
plt.xlabel(r'$x_i$')
plt.ylabel(r'$x_{i+1}$')

# Suppose we have 6 outcomes (eg. an unfair die) with the following propbabilites
p = [0.22181816, 0.16939565, 0.16688735, 0.06891783, 0.19622408, 0.17675693]

# generate the CDF
P = [np.sum(p[:i+1]) for i in range(len(p))]

plt.plot(P)
plt.xlabel('n')
plt.ylabel('P(n)')
plt.title('Cumulative Probability Distribution')

P

# Generate N random numbers sampled according to the tower, searchsorted is *fast*
N = 1000000
events = np.searchsorted(P,np.random.random(N))
plt.plot(p,'o', mec='None')
plt.hist(events, bins=len(p), normed=True, range=(-0.5,len(p) - 0.5), ec='w')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.xlim(-0.5,5.5)
plt.title('Tower Sampling')



