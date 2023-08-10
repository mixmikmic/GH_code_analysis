get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as pl
from scipy.special import hermite
from scipy.misc import factorial
from numpy import linspace, sqrt, exp, mat, zeros
from math import pi
root_pi = sqrt(pi)
def N(n, alpha):
    return sqrt(alpha / (root_pi * (2.0**n) * factorial(n)))
def phi(x,n,alpha):
    return N(n,alpha) * hermite(n)(alpha * x) * exp(-0.5 * alpha**2 * x**2)

x = linspace(-5.0,5.0,1000)
n1=0
n2=1
alpha=1.0
c1 = 1.0
c2 = 1.0
norm = sqrt(c1*c1 + c2*c2)
c1 = c1/norm
c2 = c2/norm
# Plot - I add 2.0 and 4.0 so that the graphs are offset
pl.plot(x,phi(x,n1,alpha)+0.0)
pl.plot(x,phi(x,n2,alpha)+2.0)
pl.plot(x,c1*phi(x,n1,alpha)+c2*phi(x,n2,alpha)+4.0)

from scipy.integrate import quad
quad(lambda x: (c1*phi(x,n1,alpha)+c2*phi(x,n2,alpha))*x*(c1*phi(x,n1,alpha)+c2*phi(x,n2,alpha)),-5.0,5.0)

quad(lambda x: phi(x,n1,alpha)*x*phi(x,n1,alpha),-5.0,5.0)

matsize = 6
m = mat(zeros((matsize,matsize)))
for i in range(matsize):
    for j in range(matsize):
        if i==(j+1):
            m[i,j] = sqrt(i)/(alpha*sqrt(2.0))
        elif i==(j-1):
            m[i,j] = sqrt(i+1)/(alpha*sqrt(2.0))
        else:
            m[i,j] = 0.

def printmat(A,sizei,sizej):
    for i in range(sizei):
        for j in range(sizej):
            print "%8.5f" % (A[i,j]),
        print

printmat(m,matsize,matsize)

psicol = mat(zeros((matsize,1)))
psicol[n1,0] = c1
psicol[n2,0] = c2
printmat(psicol,matsize,1)

psicol.T*m*psicol

x2 = psicol.T*(m*m)*psicol
x3 = psicol.T*(m*m*m)*psicol
x4 = psicol.T*(m*m*m*m)*psicol
print x2,x3,x4



