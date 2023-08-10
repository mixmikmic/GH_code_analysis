import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from twpca.datasets import jittered_population
from scipy.ndimage import gaussian_filter1d
rates, spikes = jittered_population()
smooth_std = 1.0
data = gaussian_filter1d(spikes, smooth_std, axis=1)

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9,3))
for ax, trial in zip(axes, data):
    ax.imshow(trial.T, aspect='auto')
    ax.set_xlabel('time')
    ax.set_ylabel('neuron')
[ax.set_title('trial {}'.format(k+1)) for k, ax in enumerate(axes)]
fig.tight_layout()

def sample_hyperparams(n):
    """Randomly draws `n` sets of hyperparameters for twPCA
    """
    n_components = np.random.randint(low=1, high=5, size=n)
    warp_scale = np.random.lognormal(mean=-6, sigma=2, size=n)
    time_scale = np.random.lognormal(mean=-6, sigma=2, size=n) 
    return n_components, warp_scale, time_scale

for param in sample_hyperparams(1000):
    pct = np.percentile(param, [0, 25, 50, 75, 100])
    print(('percentiles: ' +
          '0% = {0:.2e}, ' + 
          '25% = {1:.2e}, ' + 
          '50% = {2:.2e}, ' +
          '75% = {3:.2e}, ' + 
          '100% = {4:.2e}').format(*pct))

from twpca.crossval import hyperparam_search

# optimization parameters
fit_kw = {
    'lr': (1e-1, 1e-2),
    'niter': (100, 100),
    'progressbar': False
}

# other model options
model_kw = {
    'warptype': 'nonlinear',
    'warpinit': 'identity'
}

# search 20 sets of hyperparameters
n_components, warp_scales, time_scales = sample_hyperparams(20)

cv_results = hyperparam_search(
                data, # the full dataset
                n_components, warp_scales, time_scales, # hyperparameter arrays
                drop_prob = 0.5,
                fit_kw = fit_kw,
                model_kw = model_kw
            )

for cv_batch in cv_results['crossval_data']:
    for cv_run in cv_batch:
        plt.plot(cv_run['obj_history'])
plt.title('learning curves')

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(9,3))

axes[0].scatter(cv_results['n_components'], cv_results['mean_test'])
axes[0].set_xticks(np.unique(cv_results['n_components']))
axes[0].set_xlabel('number of components')
axes[0].set_ylabel('reconstruction error')

axes[1].set_xscale('log')
axes[1].scatter(cv_results['warp_scale'], cv_results['mean_test'])
axes[1].set_xlabel('warp regularization')
axes[1].set_title('test error on held out neurons')

axes[2].set_xscale('log')
axes[2].scatter(cv_results['time_scale'], cv_results['mean_test'])
axes[2].set_xlabel('time regularization')

axes[0].set_ylim((min(cv_results['mean_test'])*0.99, max(cv_results['mean_test'])*1.01))
fig.tight_layout()

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(9,3))

axes[0].scatter(cv_results['n_components'], cv_results['mean_train'], color='r', alpha=0.8)
axes[0].set_xticks(np.unique(cv_results['n_components']))
axes[0].set_xlabel('number of components')
axes[0].set_ylabel('reconstruction error')

axes[1].set_xscale('log')
axes[1].scatter(cv_results['warp_scale'], cv_results['mean_train'], color='r', alpha=0.8)
axes[1].set_xlabel('warp regularization')
axes[1].set_title('error during training')

axes[2].set_xscale('log')
axes[2].scatter(cv_results['time_scale'], cv_results['mean_train'], color='r', alpha=0.8)
axes[2].set_xlabel('time regularization')

axes[0].set_ylim((min(cv_results['mean_train'])*0.99, max(cv_results['mean_train'])*1.01))
fig.tight_layout()

