get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd

rng = np.random.RandomState(564)

frame = pd.read_csv('resources/up-or-down.csv')
frame.y = frame.y + rng.normal(0,.25, size=len(frame))

fig, ax = plt.subplots()
x_observed = frame.index
y_observed = frame['y'].values
ax.plot(x_observed, y_observed, label='y', color='k', linewidth=1)
ax.legend(loc='upper right', handletextpad=0)
ax.set_ylim([-3, 6])
fig.set_size_inches(10, 5)

from trcrpm import TRCRP_Mixture
model = TRCRP_Mixture(chains=8, lag=10, variables=frame.columns, rng=rng)
model.incorporate(frame)

model.resample_all(steps=500)

model.resample_hyperparameters(steps=10)

probes = range(max(model.dataset.index), max(model.dataset.index)+model.lag)
n_samples = 500

samples_ancestral = model.simulate_ancestral(probes, model.variables, n_samples)

fig, ax = plt.subplots()

# Plot the observed data.
x_observed = model.dataset.index
y_observed = model.dataset.loc[:,'y'].values
ax.plot(x_observed, y_observed, color='k', linewidth=1)

xs = frame.index
ys = frame['y'].values

# Plot the simulations.
simulations = samples_ancestral[:,:,0]
above = simulations[simulations[:,-1]>simulations[:,0]]
below = simulations[simulations[:,-1]<simulations[:,0]]

ax.plot(probes, np.median(above, axis=0), color='r',
    linestyle='--', label = 'Forecasts predicting a rise')
ax.fill_between(probes,
    np.percentile(above, 25, axis=0),
    np.percentile(above, 75, axis=0),
    color='red',
    linestyle='--',
    alpha=0.2)

ax.plot(probes, np.median(below, axis=0), color='g',
    linestyle='--', label = 'Forecasts predicting a fall')
ax.fill_between(probes,
    np.percentile(below, 25, axis=0),
    np.percentile(below, 75, axis=0),
    color='g',
    alpha=0.2)

ax.set_ylim([min(y_observed)-2, max(y_observed)+2])
ax.set_xlim([min(model.dataset.index), max(probes)+20])

# Put boxes around the similar previous lags.
ax.add_patch(Rectangle((20,0), 10, 2.5, color='cyan', linewidth=2, alpha=0.2))
ax.add_patch(Rectangle((90,0), 10, 2.5, color='cyan', linewidth=2, alpha=0.2))
ax.add_patch(Rectangle((145,0), 10, 2.5, color='cyan', linewidth=2, alpha=0.2))
ax.add_patch(Rectangle((205,0), 10, 2.5, color='cyan', linewidth=2, alpha=0.2))
ax.add_patch(Rectangle((270,0), 10, 2.5, color='cyan', linewidth=2, alpha=0.2))

# Annotate the windows.
ax.annotate('rise after window', xy=(25, 0), xytext=(25, -3), horizontalalignment='center',
    arrowprops=dict(facecolor='black', width=0.5, headwidth=5, headlength=5))
ax.annotate('fall after window', xy=(95, 0), xytext=(95, -3), horizontalalignment='center',
    arrowprops=dict(facecolor='black', width=0.5, headwidth=5, headlength=5))
ax.annotate('fall after window', xy=(150, 0), xytext=(150, -3), horizontalalignment='center',
    arrowprops=dict(facecolor='black', width=0.5, headwidth=5, headlength=5))
ax.annotate('rise after window', xy=(210, 0), xytext=(210, -3), horizontalalignment='center',
    arrowprops=dict(facecolor='black', width=0.5, headwidth=5, headlength=5))
ax.annotate('current window', xy=(275, 0), xytext=(275, -3), horizontalalignment='center',
    arrowprops=dict(facecolor='black', width=0.5, headwidth=5, headlength=5))

# Color the post-window rise and post-window fall.
ax.plot(xs[30:40], ys[30:40], linewidth=2, color='r', label='Observed rises after window')
ax.plot(xs[100:110], ys[100:110], linewidth=2, color='g', label='Observed falls after window')
ax.plot(xs[155:165], ys[155:165], linewidth=2, color='g')
ax.plot(xs[215:225], ys[215:225], linewidth=2, color='r')

# Show the forecsating region and legend.
ax.axvline(280, linestyle='--', linewidth=1, color='k')
ax.text(292, 6, 'Model\nForecasts', horizontalalignment='center')
leg = ax.legend(loc='upper left', framealpha=0, handletextpad=0)

ax.set_ylim([-4, 10])
ax.set_xlim([0, 305])

ax.set_xlabel('Timestep')
ax.grid()

fig.set_tight_layout(True)
fig.set_size_inches(10,5)
fig.savefig('resources/bimodal-posteriors.png')

samples_marginal = model.simulate(probes, model.variables, n_samples)

fig, axes = plt.subplots(nrows=model.lag/5, ncols=5)
last = model.dataset.iloc[-1].y
for lag, ax in enumerate(np.ravel(axes)):
    samples_lag = samples_marginal[:,lag,0]
    ax.hist(samples_lag[samples_lag<=last], bins=30, color='green', alpha=0.4)
    ax.hist(samples_lag[samples_lag>=last], bins=30, color='red', alpha=0.4)
    ax.set_xlim([-4,8])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_yticklabels([])
    ax.set_title('Forecast Distribution (Timestep %03d)'
        % (model.dataset.index.max() + lag), fontsize=10)
    ax.grid()
fig.set_size_inches(16,7)
fig.set_tight_layout(True)

