# Execute this cell
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=15, usetex=False)

#------------------------------------------------------------
np.random.seed(seed=42)
Nsamples=1000                                                                # Make changes here
measurements = np.random.normal(5, 1, Nsamples)
mu = np.average(measurements)

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(10, 7.5))
dist = norm(mu, 1)
dist_p = norm(5,1)
C = (1./(2.*np.pi))**(10./2.)
x = np.linspace(0, 10, 1000)
plt.plot(x, C* dist.pdf(x), c='black',label=r'$\mu_{0}=%.2f,\ \sigma=1$' % mu)
plt.plot(x, C* dist_p.pdf(x), c = 'red', label=r'$\mu = 5,\ \sigma = 1$')
#plt.axvline(x=5)

plt.xlim(0, 10)
plt.ylim(0, 0.5*C)

plt.xlabel('$\mu$')
plt.ylabel(r'$L(\{x\}|\mu,\sigma=1)$')
plt.title('Likelihood of $\mu$')

plt.legend()

import numpy as np
from sklearn.mixture import GMM
X = np.random.normal(size = (100, 1)) # 100 points in 1 dimension
model = GMM(2) # two components
model.fit(X)

model.means_  # the locations of the best fit components

