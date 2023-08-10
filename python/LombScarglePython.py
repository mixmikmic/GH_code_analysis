# Do preliminary imports and notebook setup
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

# use seaborn for plot styles
import seaborn; seaborn.set()

from astroML.datasets import fetch_LINEAR_sample
LINEAR_data = fetch_LINEAR_sample()
star_id = 10040133
t, mag, dmag = LINEAR_data.get_light_curve(star_id).T

fig, ax = plt.subplots()
ax.errorbar(t, mag, dmag, fmt='.k', ecolor='gray')
ax.set(xlabel='Time (days)', ylabel='magitude',
       title='LINEAR object {0}'.format(star_id))
ax.invert_yaxis();

from gatspy.periodic import LombScargleFast
model = LombScargleFast().fit(t, mag, dmag)
periods, power = model.periodogram_auto(nyquist_factor=100)

fig, ax = plt.subplots()
ax.plot(periods, power)
ax.set(xlim=(0.2, 1.4), ylim=(0, 0.8),
       xlabel='period (days)',
       ylabel='Lomb-Scargle Power');

# set range and find period
model.optimizer.period_range=(0.2, 1.4)
period = model.best_period
print("period = {0}".format(period))

# Compute phases of the obsevations
phase = (t / period) % 1

# Compute best-fit RR Lyrae template
from gatspy.periodic import RRLyraeTemplateModeler
model = RRLyraeTemplateModeler('r').fit(t, mag, dmag)
phase_fit = np.linspace(0, 1, 1000)
mag_fit = model.predict(period * phase_fit, period=period)

# Plot the phased data & model
fig, ax = plt.subplots()
ax.errorbar(phase, mag, dmag, fmt='.k', ecolor='gray', alpha=0.5)
ax.plot(phase_fit, mag_fit, '-k')
ax.set(xlabel='Phase', ylabel='magitude')
ax.invert_yaxis();

from scipy.signal import lombscargle

# Choose a period grid
periods = np.linspace(0.2, 1.4, 4000)
ang_freqs = 2 * np.pi / periods

# compute the (unnormalized) periodogram
# note pre-centering of y values!
power = lombscargle(t, mag - mag.mean(), ang_freqs)

# normalize the power
N = len(t)
power *= 2 / (N * mag.std() ** 2)

# plot the results
fig, ax = plt.subplots()
ax.plot(periods, power)
ax.set(ylim=(0, 0.8), xlabel='period (days)',
       ylabel='Lomb-Scargle Power');

from astroML.time_series import lomb_scargle
power = lomb_scargle(t, mag, dmag, ang_freqs)

# plot the results
fig, ax = plt.subplots()
ax.plot(periods, power)
ax.set(ylim=(0, 0.8), xlabel='period (days)',
       ylabel='Lomb-Scargle Power');

from gatspy.periodic import LombScargle

model = LombScargle(fit_offset=True).fit(t, mag, dmag)
power = model.score(periods)

# plot the results
fig, ax = plt.subplots()
ax.plot(periods, power)
ax.set(ylim=(0, 0.8), xlabel='period (days)',
       ylabel='Lomb-Scargle Power');

from gatspy.periodic import LombScargleFast

fmin = 1. / periods.max()
fmax = 1. / periods.min()
N = 10000
df = (fmax - fmin) / N

model = LombScargleFast().fit(t, mag, dmag)
power = model.score_frequency_grid(fmin, df, N)
freqs = fmin + df * np.arange(N)

# plot the results
fig, ax = plt.subplots()
ax.plot(1. / freqs, power)
ax.set(ylim=(0, 0.8), xlabel='period (days)',
       ylabel='Lomb-Scargle Power');

model = LombScargleFast().fit(t, mag, dmag)

period, power = model.periodogram_auto(nyquist_factor=200)

print("period range: ({0}, {1})".format(period.min(), period.max()))
print("number of periods: {0}".format(len(period)))

# plot the results
fig, ax = plt.subplots()
ax.plot(period, power)
ax.set(xlim=(0.2, 1.4), ylim=(0, 1.0),
       xlabel='period (days)',
       ylabel='Lomb-Scargle Power');

from scipy.signal import lombscargle as ls_scipy
from astroML.time_series import lomb_scargle as ls_astroML

def create_data(N, rseed=0, period=0.61):
    """Create noisy data"""
    rng = np.random.RandomState(rseed)
    t = 52000 + 2000 * rng.rand(N)
    dmag = 0.1 * (1 + rng.rand(N))
    mag = 15 + 0.6 * np.sin(2 * np.pi * t / period) + dmag * rng.randn(N)
    return t, mag, dmag

def compute_frequency_grid(t, oversampling=2):
    """Compute the optimal frequency grid (**not** angular frequencies)"""
    T = t.max() - t.min()
    N = len(t)
    df = 1. / (oversampling * T)
    fmax = N / (2 * T)
    return np.arange(df, fmax, df)

Nrange = 2 ** np.arange(2, 17)
t_scipy = []
t_astroML = []
t_gatspy1 = []
t_gatspy2 = []

for N in Nrange:
    t, mag, dmag = create_data(N)
    freqs = compute_frequency_grid(t)
    periods = 1 / freqs
    ang_freqs = 2 * np.pi * freqs
    f0, df, Nf = freqs[0], freqs[1] - freqs[0], len(freqs)
    
    # Don't compute the slow algorithms at very high N
    if N < 2 ** 15:
        t1 = get_ipython().magic('timeit -oq ls_scipy(t, mag - mag.mean(), ang_freqs)')
        t2 = get_ipython().magic('timeit -oq ls_astroML(t, mag, dmag, ang_freqs)')
        t3 = get_ipython().magic('timeit -oq LombScargle().fit(t, mag, dmag).score_frequency_grid(f0, df, Nf)')
        t_scipy.append(t1.best)
        t_astroML.append(t2.best)
        t_gatspy1.append(t3.best)
    else:
        t_scipy.append(np.nan)
        t_astroML.append(np.nan)
        t_gatspy1.append(np.nan)
        
    t4 = get_ipython().magic('timeit -oq LombScargleFast().fit(t, mag, dmag).score_frequency_grid(f0, df, Nf)')
    t_gatspy2.append(t4.best)

fig = plt.figure()
ax = fig.add_subplot(111, xscale='log', yscale='log')
ax.plot(Nrange, t_scipy, label='scipy: lombscargle')
ax.plot(Nrange, t_astroML, label='astroML: lomb_scargle')
ax.plot(Nrange, t_gatspy1, label='gatspy: LombScargle')
ax.plot(Nrange, t_gatspy2, label='gatspy: LombScargleFast')
ax.set(xlabel='N', ylabel='time (seconds)',
       title='Comparison of Lomb-Scargle Implementations')
ax.legend(loc='upper left');

