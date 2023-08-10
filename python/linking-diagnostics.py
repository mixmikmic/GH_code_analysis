# change the following to %matplotlib notebook for interactive plotting
get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, Series

import pims
import trackpy as tp

frames = pims.ImageSequence('../sample_data/bulk_water/*.png', as_grey=True)

f = tp.batch(frames[:4], 11, minmass=66, invert=True, meta=False)

t = tp.link_df(f, 13, memory=3, diagnostics=True)

list(t.columns)

t.head().T

remembered_frame3 = t[(t.frame == 3) & (t.diag_remembered > 0)]

print(len(remembered_frame3), 'remembered particles in frame number 3.')
remembered_frame3.head().T

subnet_particles = t[(t.frame == 1) & ~(t.diag_subnet.isnull())]

print(len(subnet_particles), 'particles involved in subnets between frames 0 and 1.')
subnet_particles.head().T

len(subnet_particles.diag_subnet.unique())

tp.annotate(subnet_particles, frames[1], plot_style={'markersize': 10});

subnets = subnet_particles.groupby('diag_subnet')[['diag_subnet_size', 'diag_subnet_iterations']].first()

# Count how many particles in the *present* frame are in each subnet,
# and include it in the results.
subnet_current_sizes = subnet_particles.diag_subnet.value_counts()
subnet_current_sizes.name = 'current_size'
subnets = subnets.join(subnet_current_sizes)

subnets.head()

plt.figure()
plt.hist(subnets.diag_subnet_iterations)
plt.xlabel('Time to solve subnet (arbitrary units)')
plt.ylabel('Number of occurrences');

