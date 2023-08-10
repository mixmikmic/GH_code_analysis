# Do imports
get_ipython().magic('matplotlib qt')

from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

# Check out the load data function. Returns file paths to load data
eegbci.load_data

# Lets get the first subjects data, the eyes closed resting block
ec_f_name = eegbci.load_data(1, [1, 2, 3])  # <- Get a list of paths to requested data

# Load some data from the first subject

ec_dat = read_raw_edf(ec_f_name[0], preload=True)
#eo_dat = read_raw_edf(ec_f_name[1], preload=True)
motor_dat = read_raw_edf(ec_f_name[2], preload=True)

# Have a quick look at the info - for eyes closed data
ec_dat.info

# Plot the eyes-closed (should open interactive viz - can scroll through)
ec_dat.plot()

ec_dat.ch_names

# Have a look at the PSD
ec_dat.plot_psd()

# Check out epoched data object
Epochs

# Pull out events
events = find_events(motor_dat, stim_channel='STI 014')

# Pick-types extracts the desired channels - here extract eeg chans, ignore stim channel, etc. 
chans = pick_types(motor_dat.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

# Settings for extracting segments of data around the events of interest
t_min, t_max = -0.5, 3
event_id = dict(hands=3, feet=3)

# Extract epochs
epochs = Epochs(motor_dat, events, event_id, t_min, t_max, proj=False, picks=chans, baseline=None, preload=True, add_eeg_ref=False)

# Average over all 
evoked = epochs.average()

get_ipython().magic('pinfo epochs')

# Look at event-related (averaged) potential - this is typically called an ERP
# In this case, perhaps most noticeable the brain seemd to get more 'positive' about 600 ms after the movement
evoked.plot()



