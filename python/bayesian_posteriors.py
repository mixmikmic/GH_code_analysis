get_ipython().magic('matplotlib inline')


import numpy as np
import func_plot_style
from scipy.stats import beta as b
from random import uniform

alpha = 3
beta = 5

theta_0 = 0.7
N = 100


plt, fig, ax = func_plot_style.subplots(figsize=(12, 8))

xmin, xmax = 0.0, 1.0
t = np.linspace(xmin, xmax, 160)

ax.set_xlim(xmin, xmax)

ax.plot(t, b.pdf(t, alpha, beta), '-', lw=2, label='prior')

x = 0
for n in range(N):
    v = int(uniform(0, 1) < theta_0)
    x += v
    alpha_n = alpha + x
    beta_n = n - x + beta
    if not n % 20 and n > 0:
        lb = r'$N={}$'.format(n)
        ax.plot(t, b.pdf(t, alpha_n, beta_n), '-', lw=2, label=lb)
    #ax.plot(t, b.pdf(t, alpha_n, beta_n), 'g-', lw=2, alpha=0.6)

ax.set_xticks((0, theta_0, 1))
ax.set_xticklabels((0, r'$\theta_0$', 1),fontsize=14)

ax.legend(loc='upper left',fontsize=14)

plt.show()





