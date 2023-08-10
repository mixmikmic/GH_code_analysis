get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot

N = 10002
r = np.zeros(N+1)
# This is my arbitrary pick (large, and *probably* primes -- haven't checked)
bigint = 6537169170218971
coef = 52361276312121

# This the infamous RANDU
coef = 65539
bigint = 2**31

#coef = 112
#bigint=2555

seed = 1.
r[0] = seed
for i in range(1,N+1):
    r[i] = (coef*r[i-1])%bigint
        
r1 = np.zeros(N/2)
r2 = np.zeros(N/2)
for i in range(0,N,2):
    r1[i/2] = float(r[i])/float(bigint)
    r2[i/2] = float(r[i+1])/float(bigint)
    
pyplot.plot(r1,r2,marker='o',linestyle='None');
    

x = np.random.random(N+1)
x1 = np.zeros(N/2)
x2 = np.zeros(N/2)
for i in range(0,N,2):
    x1[i/2] = x[i]
    x2[i/2] = x[i+1]
    
pyplot.plot(x1,x2,marker='o',linestyle='None');

from mpl_toolkits.mplot3d import Axes3D

r1 = np.zeros(N/3)
r2 = np.zeros(N/3)
r3 = np.zeros(N/3)

for i in range(0,N,3):
    r1[i/3] = r[i]/float(bigint)
    r2[i/3] = r[i+1]/float(bigint)
    r3[i/3] = r[i+2]/float(bigint)

fig = pyplot.figure()
ax = Axes3D(fig)
ax.view_init(elev=10., azim=60)

ax.scatter(r1,r2,r3,marker="o");    





