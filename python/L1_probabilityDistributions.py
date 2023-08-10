get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, poisson, binom, norm, multivariate_normal, dirichlet, gamma, beta, expon
#import scipy.stats

bernoulli.rvs(0.7, size=100)

a = np.arange(2)
colors = matplotlib.rcParams['axes.color_cycle']
plt.figure(figsize=(12,8))
for i, p in enumerate([0.1, 0.2, 0.4, 0.8]):
    ax = plt.subplot(1, 4, i+1)
    plt.bar(a, bernoulli.pmf(a, p), label=p, color=colors[i], alpha=0.5)
    ax.xaxis.set_ticks(a)
    ax.set_ylim([0,1])
    plt.legend(loc=0)
    if i == 0:
        plt.ylabel("PDF at $k$", fontsize=20)
plt.suptitle("Bernoulli probability", fontsize=20)

plt.figure(figsize=(12,8))

k = np.arange(15)
plt.figure(figsize=(12,8))
for i, lambda_ in enumerate([1, 2, 4, 6]):
    plt.plot(k, poisson.pmf(k, lambda_), '-o', label=lambda_, color=colors[i])
    #plt.fill_between(k, poisson.pmf(k, lambda_), color=colors[i], alpha=0.5)
    plt.legend()
plt.title("Poisson distribution", fontsize=20)
plt.ylabel("$p(y|\lambda)$", fontsize=20)
plt.xlabel("$y$", fontsize=20)

k = np.arange(30)
plt.figure(figsize=(12,8))
for i, lambda_ in enumerate([.1, .2, .4, .8]):
    plt.plot(k, binom.pmf(k,30, lambda_), '-o', label=lambda_, color=colors[i])
    #plt.fill_between(k, binom.pmf(k,20, lambda_), color=colors[i], alpha=0.5)
    plt.legend()
plt.title("Binomial distribution", fontsize=20)
plt.ylabel(r"$p(y|\theta)$", fontsize=20)
plt.xlabel("$y$", fontsize=20)

alpha_values = [1, 2, 4, 8,16,32]
beta_values = [1, 2, 4, 8,16,32]
x = np.linspace(1E-6, 1, 1000)

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(10, 6))

for a, b in zip(alpha_values, beta_values):
    dist = gamma(a, 0, 1./b) #1/b is a scale parameter (wee need to input scale parameter)
    plt.plot(x, beta.pdf(x, a, b), lw=2, 
             label=r'$\alpha=%.1f,\ \beta=%.1f$' % (a, b))

plt.xlim(0, 1)
plt.ylim(0, 7)

plt.xlabel('$y$', fontsize=20)
plt.ylabel(r'$p(y|\alpha,\beta)$', fontsize=20)
plt.title('Beta Distribution', fontsize=20)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

#plt.legend(loc=1, fontsize=20)
plt.show()

alpha_values = [1, 2, 4, 8,16,32]
beta_values = [1, 1, 1, 1,1,1]
x = np.linspace(1E-6, 1, 1000)

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(10, 6))

for a, b in zip(alpha_values, beta_values):
    plt.plot(x, beta.pdf(x, a, b), lw=2, 
             label=r'$\alpha=%.1f,\ \beta=%.1f$' % (a, b))

plt.xlim(0, 1)
plt.ylim(0, 7)

plt.xlabel('$y$', fontsize=20)
plt.ylabel(r'$p(y|\alpha,\beta)$', fontsize=20)
plt.title('Beta Distribution', fontsize=20)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

#plt.legend(loc=1, fontsize=20)
plt.show()

alpha_values = [1, 1, 1, 1,1,1]
beta_values = [1, 2, 4, 8,16,32]
x = np.linspace(1E-6, 1, 1000)

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(10, 6))

for a, b in zip(alpha_values, beta_values):
    plt.plot(x, beta.pdf(x, a, b), lw=2, 
             label=r'$\alpha=%.1f,\ \beta=%.1f$' % (a, b))

plt.xlim(0, 1)
plt.ylim(0, 7)

plt.xlabel('$y$', fontsize=20)
plt.ylabel(r'$p(y|\alpha,\beta)$', fontsize=20)
plt.title('Beta Distribution', fontsize=20)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

#plt.legend(loc=1, fontsize=20)
plt.show()

alpha_values = [1, 2, 3, 4]
beta_values = [1,1,1,1]
x = np.linspace(1E-6, 10, 1000)

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(10, 6))

for a, b in zip(alpha_values, beta_values):
    dist = gamma(a, 0, 1./b) #1/b is a scale parameter (wee need to input scale parameter)
    plt.plot(x, dist.pdf(x), lw=2, 
             label=r'$\alpha=%.1f,\ \beta=%.1f$' % (a, b))

plt.xlim(0, 10)
plt.ylim(0, 1.2)

plt.xlabel('$y$', fontsize=20)
plt.ylabel(r'$p(y|\alpha,\beta)$', fontsize=20)
plt.title('Gamma Distribution', fontsize=20)

plt.legend(loc=0, fontsize=20)
plt.show()

alpha_values= [3,3,3,3]
beta_values = [1,2,3,4]
x = np.linspace(1E-6, 10, 1000)

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(10, 6))

for a, b in zip(alpha_values, beta_values):
    dist = gamma(a, 0, 1./b)
    plt.plot(x, dist.pdf(x), lw=2, 
             label=r'$\alpha=%.1f,\ \beta=%.1f$' % (a, b))
plt.xlim(0, 10)
plt.ylim(0, 1.2)

plt.xlabel('$y$',fontsize=20)
plt.ylabel(r'$p(y|\alpha,\beta)$',fontsize=20)
plt.title('Gamma Distribution',fontsize=20)

plt.legend(loc=0,fontsize=20)
plt.show()

# plot the distributions
lambda_values= [0.5, 1, 2, 4]
x = np.linspace(1E-6, 10, 1000)

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(10, 6))

for l in lambda_values:
    plt.plot(x, expon.pdf(x, scale=1./l), lw=2, 
             label = "$\lambda = %.1f$"%l)
plt.xlim(0, 10)
plt.ylim(0, 1.2)

plt.xlabel('$y$',fontsize=20)
plt.ylabel(r'$p(y|\lambda)$',fontsize=20)
plt.title('Exponential Distribution',fontsize=20)

plt.legend(loc=0,fontsize=20)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#fig, ax = plt.subplots(figsize=(10, 6))
fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
x, y = np.mgrid[-5:5:.05, -5:5:.05]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
z = rv.pdf(pos)
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(0, .5)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.xlabel('$y_1$',fontsize=20)
plt.ylabel(r'$y_2$',fontsize=20)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

np.set_printoptions(precision=2)

def stats(scale_factor, alpha=[1, 1, 1], N=10000):
    samples = dirichlet(alpha = scale_factor * np.array(alpha)).rvs(N)
    print ("                          alpha:", scale_factor)
    print ("              element-wise mean:", samples.mean(axis=0))
    print ("element-wise standard deviation:", samples.std(axis=0))
    print ()
    
for scale in [0.1, 1, 10, 100, 1000]:
    stats(scale)



