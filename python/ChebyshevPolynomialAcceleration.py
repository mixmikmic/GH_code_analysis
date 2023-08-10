#math and linear algebra stuff
import numpy as np
import numpy.linalg as la
import scipy as sc

#plots
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15.0, 15.0)
#mpl.rc('text', usetex = True)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

"""
Draw all Chebyshev Polynomial up to order 5 over
the range [-1,1]
"""

from math import factorial
from scipy import misc as misc

def comb(n, k):
    return misc.comb(n,k)
    #return factorial(n) / (factorial(k) * factorial(n - k))

ChebyOrderMax = 5
cm = plt.get_cmap('gist_rainbow')
colors = [cm(1.*i/ChebyOrderMax) for i in range(ChebyOrderMax+1)]
plt.figure(figsize=(8,8))
plt.title( "First "+str(ChebyOrderMax)+" order Chebyshev polynomials")
x = np.linspace(-1,1,500)
plt.xlabel("x")
plt.ylabel("y")
ax = plt.subplot()
ax.set_ylim([-1.1,1.1])

def Cheby(x, order):
    y=[comb(order,2*i)*
       x**(order-2*i)*(x**2-1)**i
       for i in range(order//2+1)]
    return np.sum(y,axis=0)

for color,order in zip(colors, range(ChebyOrderMax+1)):
    plt.plot(x,Cheby(x,order),color=color,label="Order "+str(order))

plt.legend()

"""
Check numerically that recursion relation is valid
"""
ChebyOrderMax = 1
x = np.linspace(-1,1,500)
for order in range(ChebyOrderMax+1):
     assert(np.allclose(Cheby(x,order+2),2*x*Cheby(x,order+1)-Cheby(x,order)))
    



