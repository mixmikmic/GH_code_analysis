import math
from matplotlib import pylab as plt
get_ipython().magic('matplotlib inline')

def posion_dist(m, k):
    return (m**k * math.e**(-m)) / math.factorial(k)

posion_dist(2, 3)

N = range(1, 20)
M = 7
plt.plot(N, [posion_dist(M, i) for i in N])

