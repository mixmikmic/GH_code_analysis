get_ipython().run_line_magic('matplotlib', 'inline')

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# from mayavi import mlab

import mne
from mne.viz import plot_trans, snapshot_brain_montage

print(__doc__)

mat = loadmat(mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat')
ch_names = mat['ch_names'].tolist()
elec = mat['elec']
dig_ch_pos = dict(zip(ch_names, elec))
mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
print('Created %s channel positions' % len(ch_names))

mon, dig_ch_pos
elec.shape, len(dig_ch_pos), mon

info = mne.create_info(ch_names, 1000., 'ecog', montage=mon)

subjects_dir = mne.datasets.sample.data_path() + '/subjects'
fig = plot_trans(info, trans=None, subject='sample', subjects_dir=subjects_dir)
mlab.view(200, 70)

# We'll once again plot the surface, then take a snapshot.
fig = plot_trans(info, trans=None, subject='sample', subjects_dir=subjects_dir)
mlab.view(200, 70)
xy, im = snapshot_brain_montage(fig, mon)

# Convert from a dictionary to array to plot
xy_pts = np.vstack(xy[ch] for ch in info['ch_names'])

# Define an arbitrary "activity" pattern for viz
activity = np.linspace(100, 200, xy_pts.shape[0])

# This allows us to use matplotlib to create arbitrary 2d scatterplots
_, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im)
ax.scatter(*xy_pts.T, c=activity, s=200, cmap='coolwarm')
ax.set_axis_off()
plt.show()

