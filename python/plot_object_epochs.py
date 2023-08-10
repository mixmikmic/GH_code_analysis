get_ipython().run_line_magic('matplotlib', 'inline')

import mne
import os.path as op
import numpy as np
from matplotlib import pyplot as plt

data_path = mne.datasets.sample.data_path()
# Load a dataset that contains events
raw = mne.io.read_raw_fif(
    op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif'))

# If your raw object has a stim channel, you can construct an event array
# easily
events = mne.find_events(raw, stim_channel='STI 014')

# Show the number of events (number of rows)
print('Number of events:', len(events))

# Show all unique event codes (3rd column)
print('Unique event codes:', np.unique(events[:, 2]))

# Specify event codes of interest with descriptive labels.
# This dataset also has visual left (3) and right (4) events, but
# to save time and memory we'll just look at the auditory conditions
# for now.
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}

epochs = mne.Epochs(raw, events, event_id, tmin=-0.1, tmax=1,
                    baseline=(None, 0), preload=True)
print(epochs)

print(epochs.events[:3])
print(epochs.event_id)

print(epochs[1:5])
print(epochs['Auditory/Right'])

# These will be epochs objects
for i in range(3):
    print(epochs[i])

# These will be arrays
for ep in epochs[:2]:
    print(ep)

epochs.drop([0], reason='User reason')
epochs.drop_bad(reject=dict(grad=2500e-13, mag=4e-12, eog=200e-6), flat=None)
print(epochs.drop_log)
epochs.plot_drop_log()
print('Selection from original events:\n%s' % epochs.selection)
print('Removed events (from numpy setdiff1d):\n%s'
      % (np.setdiff1d(np.arange(len(events)), epochs.selection).tolist(),))
print('Removed events (from list comprehension -- should match!):\n%s'
      % ([li for li, log in enumerate(epochs.drop_log) if len(log) > 0]))

epochs_fname = op.join(data_path, 'MEG', 'sample', 'sample-epo.fif')
epochs.save(epochs_fname)

epochs = mne.read_epochs(epochs_fname, preload=False)

ev_left = epochs['Auditory/Left'].average()
ev_right = epochs['Auditory/Right'].average()

f, axs = plt.subplots(3, 2, figsize=(10, 5))
_ = f.suptitle('Left / Right auditory', fontsize=20)
_ = ev_left.plot(axes=axs[:, 0], show=False)
_ = ev_right.plot(axes=axs[:, 1], show=False)
plt.tight_layout()

