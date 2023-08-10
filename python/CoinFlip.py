import warnings
import numpy as np
import pandas as pd

from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns

import pymc3 as pm
import scipy.stats as stats

sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
np.random.seed(42)

alpha = 3
beta = 3

fig, ax = plt.subplots(1, 1)
prior_beta = stats.beta(alpha, beta)
x = np.linspace(0, 1, 100)
ax.plot(x, prior_beta.pdf(x))
plt.title('Beta prior with alpha = beta = 3')
plt.xlabel('x')
plt.xlim([0, 1])
plt.ylim([0, 2.5])
plt.ylabel('pdf')
plt.show()

n = 100  # Tosses
h = 62  # Number of times head

niter = 2000

with pm.Model() as model: # context management
    # define priors
    p = pm.Beta('p', alpha=alpha, beta=beta, testval=0.5)
    # define likelihood
    y = pm.Binomial('y', n=n, p=p, observed=h)

# Run inference
with model:
    step = pm.Metropolis() # Have a choice of samplers
    trace = pm.sample(niter, step, random_seed=42, progressbar=True)

pm.traceplot(trace)
plt.show()

print('p trace: ', trace['p'].shape)
p_trace_burnin = trace['p'][500:]
print('p_trace_burnin: ', p_trace_burnin.shape)

# Compare prior and posterior
sns.distplot(p_trace_burnin, label='post')
x = np.linspace(0, 1, 100)
plt.plot(x, stats.beta.pdf(x, alpha, beta), label='prior')
plt.legend(loc='best')
plt.title('Posterior vs prior distribution.')
plt.ylabel('Frequency')
plt.xlim([0, 1])
plt.show()

hpd = pm.stats.hpd(p_trace_burnin, alpha=0.05)
print('5% HPD: {}'.format(hpd))

pm.plot_posterior(p_trace_burnin, ref_val=0.5, point_estimate='mean')
plt.show()

