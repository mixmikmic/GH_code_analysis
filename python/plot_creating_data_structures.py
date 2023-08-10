get_ipython().run_line_magic('matplotlib', 'inline')

from __future__ import print_function

import mne
import numpy as np

# Create some dummy metadata
n_channels = 32
sampling_rate = 200
info = mne.create_info(n_channels, sampling_rate)
print(info)

# Names for each channel
channel_names = ['MEG1', 'MEG2', 'Cz', 'Pz', 'EOG']

# The type (mag, grad, eeg, eog, misc, ...) of each channel
channel_types = ['grad', 'grad', 'eeg', 'eeg', 'eog']

# The sampling rate of the recording
sfreq = 1000  # in Hertz

# The EEG channels use the standard naming strategy.
# By supplying the 'montage' parameter, approximate locations
# will be added for them
montage = 'standard_1005'

# Initialize required fields
info = mne.create_info(channel_names, sfreq, channel_types, montage)

# Add some more information
info['description'] = 'My custom dataset'
info['bads'] = ['Pz']  # Names of bad channels

print(info)

# Generate some random data
data = np.random.randn(5, 1000)

# Initialize an info structure
info = mne.create_info(
    ch_names=['MEG1', 'MEG2', 'EEG1', 'EEG2', 'EOG'],
    ch_types=['grad', 'grad', 'eeg', 'eeg', 'eog'],
    sfreq=100
)

custom_raw = mne.io.RawArray(data, info)
print(custom_raw)

# Generate some random data: 10 epochs, 5 channels, 2 seconds per epoch
sfreq = 100
data = np.random.randn(10, 5, sfreq * 2)

# Initialize an info structure
info = mne.create_info(
    ch_names=['MEG1', 'MEG2', 'EEG1', 'EEG2', 'EOG'],
    ch_types=['grad', 'grad', 'eeg', 'eeg', 'eog'],
    sfreq=sfreq
)

# Create an event matrix: 10 events with alternating event codes
events = np.array([
    [0, 0, 1],
    [1, 0, 2],
    [2, 0, 1],
    [3, 0, 2],
    [4, 0, 1],
    [5, 0, 2],
    [6, 0, 1],
    [7, 0, 2],
    [8, 0, 1],
    [9, 0, 2],
])

event_id = dict(smiling=1, frowning=2)

# Trials were cut from -0.1 to 1.0 seconds
tmin = -0.1

custom_epochs = mne.EpochsArray(data, info, events, tmin, event_id)

print(custom_epochs)

# We can treat the epochs object as we would any other
_ = custom_epochs['smiling'].average().plot()

# The averaged data
data_evoked = data.mean(0)

# The number of epochs that were averaged
nave = data.shape[0]

# A comment to describe to evoked (usually the condition name)
comment = "Smiley faces"

# Create the Evoked object
evoked_array = mne.EvokedArray(data_evoked, info, tmin,
                               comment=comment, nave=nave)
print(evoked_array)
_ = evoked_array.plot()

