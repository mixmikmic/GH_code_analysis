import numpy as np
import matplotlib.pyplot as plt

n = 10000
sim = np.random.rand(n,)
inside = len(sim[(sim>0.7)&(sim<0.75)])
print("Percentage = %d / %d = %0.4f"%(inside, n, inside/n))

print(1.0/np.exp(2))

tau = 1
n = 10000
sim = np.random.exponential(scale=tau, size=(n,))
inside = len(sim[sim>2*tau])
print("Percentage = %d / %d = %0.4f"%(inside, n, inside/n))

from scipy.special import erf
print((1-erf(np.sqrt(2)))/2)

n = 10000
mu = 0
sigma = 3
sim = sigma*np.random.randn(n,) + mu
inside = len(sim[sim>2*sigma])
print("Percentage = %d / %d = %0.4f"%(inside, n, inside/n))

import seaborn as sns

x1 = np.random.rand(10000,)
x2 = np.random.rand(10000,)

x = x1 + x2

fig = plt.figure(figsize=(12,4))
ax1, ax2, ax3 = [fig.add_subplot(1,3,i+1) for i in range(3)]

ax1.hist(x1, normed=True, bins=25)
ax2.hist(x2, normed=True, bins=25)
ax3.hist(x,  normed=True, bins=25)

ax1.set_xlabel(r'$x_1$')
ax2.set_xlabel(r'$x_2$')
ax3.set_xlabel(r'$x_1 + x_2$')

plt.show()

def draw_uniform_sums(nx):

    tot = 10000

    xs = []
    xsum = np.zeros(tot,)
    for x in range(nx):
        x = np.random.rand(tot,)
        xsum += x
        xs.append(x)

    fig = plt.figure(figsize=(14,2))
    axes = [fig.add_subplot(1,nx,i+1) for i in range(nx)]

    for i, (ax, x) in enumerate(zip(axes,xs)):
        ax.hist(x, normed=True, bins=25)
        ax.set_xlabel(r'$x_%d$'%(i+1))

    fig2 = plt.figure(figsize=(4,4))
    ax2 = fig2.add_subplot(1,1,1)
    ax2.hist(xsum, normed=True, bins=25)
    ax2.set_xlabel(r'$\Sigma x_i$')
    
    plt.show()

draw_uniform_sums(3)

draw_uniform_sums(6)

draw_uniform_sums(8)



