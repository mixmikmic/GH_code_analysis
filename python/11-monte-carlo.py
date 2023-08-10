get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn

x = 1
u = np.zeros(1000)
a, b, M = 1103515245, 12345, 2**31

for i in xrange(len(u)):
    x_new = (a*x + b) % M
    u[i] = float(x_new) / M
    x = x_new
    
plt.figure(figsize=(8, 4))
plt.subplot(1,2,1)
# you can also use plt.hist(u),
# here I just want to make it more beatiful.
seaborn.distplot(u, kde=False, norm_hist=True)
plt.title('Our generator')

plt.subplot(1,2,2)
np.random.seed(1)
seaborn.distplot(np.random.random(1000), kde=False, norm_hist=True)
plt.title('np.random.random()')

# plots 1000 standard normal samples into bins.
seaborn.distplot(np.random.standard_normal(1000));
plt.xlim([-4,4]);

import matplotlib.mlab as mlab

dx = .01
x = np.arange(-4, 4, dx)

# probility density function of N(0, 1)
pdf = mlab.normpdf(x, 0, 1.0)

# cumulative density function
cdf = np.cumsum(pdf*dx)

# plot the curves
plt.plot(x, pdf, label='pdf')
plt.plot(x, cdf, label='cdf')
plt.annotate('', xy=(0.8, 0.8), xytext=(-4, 0.8),
            arrowprops=dict(color='orange', shrink=0.01, width=1, headwidth=6))
plt.annotate('', xy=(0.9, 0), xytext=(0.9, 0.8),
            arrowprops=dict(color='orange', shrink=0.01, width=1, headwidth=6))
plt.ylabel('$u$')
plt.xlabel('$x$')
plt.legend(loc='best');

dx = .01
x = np.arange(0, 6, dx)

pdf = np.exp(-x)
seaborn.distplot(-np.log(1 - np.random.random(10000)), kde=False, norm_hist=True, label="samples");
plt.plot(x, pdf, label='exponential pdf')
plt.title('Expoential distribution')
plt.legend()
plt.ylabel('density')
plt.xlim([0, 6])
plt.xlabel('$x$');

# draw points
true = np.random.standard_normal(1000)
true_x, true_y = true[:500], true[500:]
plt.scatter(true_x, true_y, alpha=0.5)

# draw r and theta
theta = np.linspace(0, 2*np.pi, 100)
r, r2 = 1.6, 2
plt.plot(r*np.cos(theta), r*np.sin(theta), label='r')
plt.plot(r2*np.cos(theta), r2*np.sin(theta), label='r+dr')

x = np.array([0, 3])
t1, t2 = np.pi/6, 1.5*np.pi/6
plt.plot(x, t1*x, label='theta')
plt.plot(x, t2*x, label='theta+dtheta')

plt.legend()
plt.axis('equal');

# Generator
theta = 2 * np.pi * np.random.random(2000)
r = np.sqrt(-2 * np.log(np.random.random(2000)))

samples = np.hstack([r*np.cos(theta), r*np.sin(theta)])
    
dx = .01
x = np.arange(-4, 4, dx)

# probility density function of N(0, 1)
pdf = mlab.normpdf(x, 0, 1.0)
seaborn.distplot(samples, kde=False, norm_hist=True, label="samples");
plt.plot(x, pdf, label='normal pdf')
plt.title('Normal distribution')
plt.legend()
plt.ylabel('density')
plt.xlim([-4, 4])
plt.xlabel('$x$');

dx = .01
x = np.arange(0, 6, dx)

M = 1

def f(x):
    return (1/np.sqrt(2*np.pi)) * np.exp(-np.power(x, 2) / 2)

def g(x):
    return np.exp(-x)

plt.plot(x, f(x), label='$f(x)$', color='green')
plt.plot(x, M*g(x), label='Mg(x)', color='red')
plt.fill_between(x, f(x), 0, color='green', alpha=0.2, label='accept')
plt.fill_between(x, M*g(x), f(x), color='red', alpha=0.2, label='reject')
plt.title('Rejection sampling')
plt.legend()
plt.ylabel('density')
plt.xlim([0, 6])
plt.xlabel('$x$');

# Rejection sampling,
x = -np.log(1 - np.random.random(4000)) # Sample x from g(x), exp dist.
u = np.random.random(4000)              # generate u from unif(0,1).
keep = u < (f(x) / (M*g(x)))            # accept if f(x) / Mg(x).

# Flip the sign of half accepted examples.
samples = x[keep]
samples[:len(samples)/2] = -samples[:len(samples)/2] 

# Plot histogram of samples and standard normal pdf.
lx = np.arange(-4, 4, 0.1)
plt.plot(lx, mlab.normpdf(lx, 0, 1.0), label='normal pdf')
seaborn.distplot(samples, kde=False, norm_hist=True, label="samples");
plt.legend();



