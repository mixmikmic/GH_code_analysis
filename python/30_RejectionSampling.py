import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
plt.style.use('notebook');
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
colors = ["#2078B5", "#FF7F0F", "#2CA12C", "#D72827", "#9467BE", "#8C574B",
            "#E478C2", "#808080", "#BCBE20", "#17BED0", "#AEC8E9", "#FFBC79", 
            "#98E08B", "#FF9896", "#C6B1D6", "#C59D94", "#F8B7D3", "#C8C8C8", 
           "#DCDC8E", "#9EDAE6"]

def p(y,λ):
    return λ*np.exp(-λ*y)

N = 100000
λ = 0.5
y = np.linspace(0,100,N)
plt.plot(y,p(y,λ),color=colors[0], label=r'$%3.1f\mathrm{e}^{-%3.1fy}$'%(λ,λ))

# sample y from a uniform x
x = np.random.random(N)
sampled_y = -(1/λ)*np.log(1.0-x)

plt.hist(sampled_y, bins=100, normed=True, ec='w', label='sampled', fc=colors[0], alpha=0.2);
plt.xlim(0,10);
plt.xlabel('y')
plt.ylabel('p(y)')
plt.title('Exponential Distribution')
plt.legend(loc='upper right')

from scipy.constants import pi as π
N = 100000

# our uniform random numbers
x1 = np.random.random(N)
x2 = np.random.random(N)

# generate the Box-Muller values
x = np.sqrt(-2.0*np.log(1-x1))*np.cos(2.0*π*x2)
y = np.sqrt(-2.0*np.log(1-x1))*np.sin(2.0*π*x2)

# combine them into 1 array
r = np.hstack([x,y])

# produce a plot comparing with the actual distribution
px = np.linspace(-4,4,1000)
plt.plot(px,np.exp(-px**2/2)/np.sqrt(2*π), color=colors[0], label=r'$\frac{1}{\sqrt{2\pi}}\mathrm{e}^{-y^2/2}$')
plt.hist(r, bins=100, normed=True, ec='w', label='sampled',fc=colors[0], alpha=0.2)
plt.xlabel('y')
plt.ylabel('p(y)')
plt.xlim(-4,4)
plt.title('Box-Muller')
plt.legend(loc='upper left')

from scipy import integrate
xmin,xmax = -5,5

f = lambda x: 1.0/np.sqrt(np.cosh(x))
A = 1.0/integrate.quad(f, xmin, xmax)[0]
print(A)

p = lambda x: A*f(x)

# Let's start by plotting
px = np.linspace(xmin,xmax,10000)
plt.plot(px,p(px), color=colors[0], label=r'$\frac{%4.2f}{\sqrt{\cosh(x)}}$'%A)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.xlim(-5,5);
plt.legend()

print('p(0) = ',p(0))

from scipy.optimize import minimize
pmax = -minimize(lambda x: -p(x),-5).fun
print('max p(x) = ',pmax)

N = 10**6
x = xmin + (xmax-xmin)*np.random.random(N)
y = p(0)*np.random.random(N)

#accepted = np.array([])
#for i in range(N):
#    if y[i] < p(x[i]):
#        accepted = np.append(accepted,x[i])
        
accepted = x[y < p(x)]

plt.plot(px,p(px),color=colors[0], label=r'$\frac{%4.2f}{\sqrt{\cosh(x)}}$'%A)
plt.hist(accepted, bins=100, normed=True, ec='w', label='sampled', fc=colors[0], alpha=0.2)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.xlim(-5,5);
plt.legend()

x_accepted = x[y < p(x)]
y_accepted = y[y < p(x)]

x_rejected = x[y >= p(x)]
y_rejected= y[y >= p(x)]

plt.axhline(y=pmax, color='gray', lw=2, label='p(0)')
plt.plot(px,p(px),color=colors[0], label=r'$\frac{%4.2f}{\sqrt{\cosh(x)}}$'%A)

plt.plot(x_accepted,y_accepted,'o', mfc='None', mec=colors[0], ms=3, label='accepted')
plt.plot(x_rejected,y_rejected,'x', mfc='None', mec=colors[1], ms=3, label='rejected')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.xlim(-5,5);
plt.legend(loc='upper right', frameon=True, framealpha=0.9)
#plt.savefig('data/rejection.svg')



