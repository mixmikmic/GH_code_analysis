import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
plt.style.use('notebook');
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
colors = ["#2078B5", "#FF7F0F", "#2CA12C", "#D72827", "#9467BE", "#8C574B",
            "#E478C2", "#808080", "#BCBE20", "#17BED0", "#AEC8E9", "#FFBC79", 
            "#98E08B", "#FF9896", "#C6B1D6", "#C59D94", "#F8B7D3", "#C8C8C8", 
           "#DCDC8E", "#9EDAE6"]

def simpsons_rule(f,x,*params):
    '''The trapezoidal rule for numerical integration of f(x) over x.'''
    a,b = x[0],x[-1]
    Δx = x[1] - x[0]
    N = x.size
    
    #I = (f(a,*params) + f(b,*params))/3.0
    #I += (4.0/3.0)*np.sum([f(a + j*Δx,*params) for j in range(1,N,2)])
    #I += (2.0/3.0)*np.sum([f(a + j*Δx,*params) for j in range(2,N,2)])
    
    I = (f(a,*params) + f(b,*params))/3.0
    I += (4.0/3.0)*sum([f(a+i*Δx,*params) for i in range(1,N,2)])
    I += (2.0/3.0)*sum([f(a+i*Δx,*params) for i in range(2,N,2)])
    
    return Δx*I

from scipy.constants import pi as  π
from scipy.special import erf

def erf_kernel(t):
    '''The error function kernel.'''
    return  (2.0/np.sqrt(π))*np.exp(-t*t)

Δx = 0.001
x = np.linspace(0,1,20)
erf_approx = np.zeros_like(x)

for j,cx in enumerate(x[1:]):
    N = int(cx/Δx)
    if N % 2: N += 1
    x_int = np.linspace(0,cx,N)
    erf_approx[j+1] = simpsons_rule(erf_kernel,x_int)

# plot the results and compare with the 'exact' value
plt.plot(x,erf_approx,'o', mec=colors[0], mfc=colors[0], mew=1, ms=8, label="Simpson's Rule")
plt.plot(x,erf(x), color=colors[1],zorder=0, label='scipy.special.erf')
plt.legend(loc='lower right')
plt.xlabel('x')
plt.ylabel('erf(x)')

def f(x):
    return 1.0/np.sqrt(x**4 + 1)

def g(y):
    return 1.0/np.sqrt(y**4 + (1-y)**4)

y = np.linspace(0,1,100000)
print(simpsons_rule(g,y))

from scipy import integrate
print(integrate.quad(f, 0, np.inf))



