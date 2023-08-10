get_ipython().magic('matplotlib inline')

# %load ../../norm_den_seq.py
import numpy as np
import func_plot_style
from scipy.stats import norm 

mu0 = 0
sigma0 = 1
a = 0.8
b = 1
c = 1

plt, fig, ax = func_plot_style.subplots()

xmin, xmax = -4.0, 8.0
x = np.linspace(xmin, xmax, 100)

mu = mu0
sigma2 = sigma0**2

for t in range(8):
    f = lambda x: norm.pdf(x, mu, np.sqrt(sigma2))
    lb = r'$t={}$'.format(t)
    ax.plot(x, f(x), '-', lw=2.6, label=lb)
    mu = b + a * mu
    sigma2 = a * a * sigma2 + c**2

ax.legend(loc='upper left')
plt.show()


