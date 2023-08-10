from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt

# change the following to %matplotlib notebook for interactive plotting
get_ipython().magic('matplotlib inline')

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

frames = pims.ImageSequence('../sample_data/bulk_water/*.png', as_grey=True)

frames

print(frames[0])  # the first frame

frames[0]

plt.imshow(frames[0]);

frames[123].frame_no

frames[123].metadata  # Scientific formats can pass experiment meta data here.

f = tp.locate(frames[0], 11, invert=True)

f.head() # shows the first few rows of data

plt.figure()  # make a new figure
tp.annotate(f, frames[0]);

fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)

# Optionally, label the axes.
ax.set(xlabel='mass', ylabel='count');

f = tp.locate(frames[0], 11, invert=True, minmass=200)

plt.figure()
tp.annotate(f, frames[0]);

plt.figure()
tp.subpx_bias(f);

plt.figure()
tp.subpx_bias(tp.locate(frames[0], 7, invert=True, minmass=200));

f = tp.batch(frames[:300], 11, minmass=200, invert=True);

t = tp.link_df(f, 5, memory=3)

t.head()

t1 = tp.filter_stubs(t, 50)
# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())

plt.figure()
tp.mass_size(t1.groupby('particle').mean()); # convenience function -- just plots size vs. mass

t2 = t1[((t1['mass'] > 250) & (t1['size'] < 3.0) &
         (t1['ecc'] < 0.1))]

plt.figure()
tp.annotate(t2[t2['frame'] == 0], frames[0]);

plt.figure()
tp.plot_traj(t1);

d = tp.compute_drift(t1)

plt.figure()
d.plot();

tm = tp.subtract_drift(t1.copy(), d)

plt.figure()
ax = tp.plot_traj(tm)

im = tp.imsd(tm, 100/285., 24)  # microns per pixel = 100/285., frames per second = 24

fig, ax = plt.subplots()
ax.plot(im.index, im, 'k-', alpha=0.1)  # black lines, semitransparent
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
ax.set_xscale('log')
ax.set_yscale('log')

em = tp.emsd(tm, 100/285., 24)

fig, ax = plt.subplots()
ax.plot(em.index, em, 'o')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
ax.set(ylim=(1e-2, 10));

plt.figure()
plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
plt.xlabel('lag time $t$');
tp.utils.fit_powerlaw(em)  # performs linear best fit in log space, plots

with tp.PandasHDFStore('data.h5') as s:
    tp.batch(frames, 11, invert=True, minmass=200, output=s)

with tp.PandasHDFStore('data.h5') as s:
    for linked in tp.link_df_iter(s, 5, memory=3):
        s.put(linked)

with tp.PandasHDFStore('data.h5') as s:
    trajectories = pd.concat(iter(s))

get_ipython().magic("timeit tp.batch(frames[:10], 11, invert=True, minmass=200, engine='numba')")

get_ipython().magic("timeit tp.batch(frames[:10], 11, invert=True, minmass=200, engine='python')")

