import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA, PCA
from modelingtools import delay_time_series, plot_gabors_2d
from sklearn.preprocessing import scale
from numpy.fft import fft2, fftshift, fftfreq

import warnings
warnings.simplefilter('ignore')
get_ipython().magic('matplotlib inline')

raw_path = '../raw_data/'
save_path = '../features/'

# We'll save all timeseries data at this sampling frequency
sfreq_new = 200

# We'll use inline notebook plotting for interactivity here
# %matplotlib notebook

ecog = mne.read_epochs(raw_path + 'ecog-epo.fif', preload=True)
_ = ecog.plot(scalings='auto', n_epochs=5, n_channels=10)

# Copy our raw data so it isn't overwritten, then rescale it to baseline.
ecog_av = ecog.copy()
mne.baseline.rescale(ecog_av._data, ecog.times, (-.8, -.1),
                     mode='zscore', copy=False)

# Average, crop, then plot the data
_ = ecog_av.average().crop(-.2, .5).plot()

get_ipython().magic('matplotlib inline')
# First we'll visualize a morlet wavelet:
fig, ax = plt.subplots()
wavelet_freqs=  [3, 6, 10, 16, 32]
cmap = plt.cm.rainbow
for ii, freq in enumerate(wavelet_freqs):
    wavelet = mne.time_frequency.morlet(ecog.info['sfreq'], [freq])[0]
    time = np.arange(len(wavelet)) / float(ecog.info['sfreq'])
    time -= time[len(time) // 2]
    color = cmap(float(ii) / len(wavelet_freqs))
    ax.plot(time, wavelet + .5 * ii, c=color)
    ax.text(-.25, .5 * ii + .25, '{}Hz'.format(freq), horizontalalignment='right')
    ax.set_xlim([-1, 1])
_ = plt.setp(ax.get_yticklabels(), visible=False)
ax.set_xlabel('Time (s)')
ax.set_title('Sample Wavelets')

high_freq = ecog.copy()
freqs = np.logspace(np.log10(70), np.log10(140), 14)
etfr = mne.time_frequency.tfr_morlet(
    high_freq, freqs, picks=range(len(high_freq.ch_names)),
    n_cycles=freqs / 4., average=False, use_fft=True, return_itc=False)

# Now average across frequencies for a single time-varying amplitude
tfrs = etfr.data.mean(axis=2)

# We'll rescale the data so that it's in standard units above a baseline
high_freq._data = tfrs.copy()
_ = mne.baseline.rescale(high_freq._data, high_freq.times, (-.8, -.1),
                         mode='zscore', copy=False)

# Now plot
_ = high_freq.copy().crop(-.5, 1.5).average().plot()

# Now, resample to save computation time / space
high_freq.resample(sfreq_new, npad='auto')

# Save to disk
high_freq.save(save_path + 'hfa_ecog-epo.fif')

# If you'd like to use the following interactively, run this line
# Don't forget to run `%matplotlib inline` aftward
# %matplotlib notebook

# We've already split our auditory data into a trials format, called Epochs
audio = mne.read_epochs(raw_path + 'audio-epo.fif', preload=True)

# We'll add a minimal amount of white noise so that we don't have zeros
audio._data += .1 * np.std(audio._data) * np.random.randn(*audio._data.shape)

# Now visualize
_ = audio.plot(picks=[0], scalings='auto', n_epochs=5)

get_ipython().magic('matplotlib inline')
# Take the absolute value, then low-pass filter
envelope = audio.copy()
envelope._data = np.abs(envelope._data)
envelope._data = mne.filter.filter_data(envelope._data,
                                        envelope.info['sfreq'],
                                        None,
                                        10, filter_length='1s')
envelope.resample(sfreq_new)

_ = envelope.plot(picks=[0], scalings='auto', n_epochs=5)

envelope.save(save_path + 'envelope-epo.fif')

# We'll use log-spaced frequencies
frequencies = np.logspace(np.log10(300), np.log10(8000), 64)
n_decim = int(audio.info['sfreq'] // sfreq_new)

high_freq = ecog.copy()
freqs = np.logspace(np.log10(70), np.log10(140), 14)
tfrs = mne.time_frequency.tfr_morlet(
    audio, frequencies, picks=[0], n_cycles=frequencies / 4., average=False,
    use_fft=True, return_itc=False, decim=n_decim)

fig, ax = plt.subplots()

mask = mne.utils._time_mask(tfrs.times, -.5, 4)
plt_tfrs = tfrs.data[1].squeeze()[..., mask]
mesh = ax.pcolormesh(tfrs.times[mask], frequencies, np.log(plt_tfrs),
                     cmap='viridis', vmin=10, vmax=20)
ax.axis('tight')
ax.set_title('Spectrogram of speech')
ax.set_ylabel('Frequency Band (Hz)')
ax.set_xlabel('Time (s)')
cbar = fig.colorbar(mesh)
cbar.set_label('Log Amplitude')

# Now convert it to Epochs so we can use later
info_spec = mne.create_info(['{:.2f}'.format(ifreq) for ifreq in frequencies],
                            sfreq_new, 'misc')
tfrs = mne.EpochsArray(tfrs.data.squeeze(), info_spec, tmin=tfrs.times.min())

tfrs.save(save_path + 'spectrogram-epo.fif')

# Read in timing information
stimuli = pd.read_csv(raw_path + 'time_info.csv', index_col=0)
sfreq_timit = pd.read_csv(raw_path + 'meta_sub.csv')['sfreq_timit'].values[0]

# Loop through each phoneme, find its time onsets, and fill in 1s
phonemes = []
for ii, stim_name in enumerate(stimuli['stim_name']):
    stim_name = stim_name.replace('.wav', '')
    with open(raw_path + 'timit_word_info/text_info/{}.PHN'.format(stim_name, 'r')) as f:
        ph_sentence = f.readlines()
        ph_sentence = [jj.replace('\n', '').split(' ') for jj in ph_sentence]
        ph_sentence = pd.DataFrame(ph_sentence)
        ph_sentence['number'] = ii
        phonemes.append(ph_sentence)
phonemes = pd.concat(phonemes)
phonemes.columns = ['start', 'stop', 'phoneme', 'number']

# TIMIT timings are given in samples. Here we divide by sfreq to get seconds.
phonemes[['start', 'stop']] = phonemes[['start', 'stop']].apply(pd.to_numeric)
phonemes[['start', 'stop']] /= sfreq_timit

unique_phonemes = phonemes['phoneme'].unique()
phoneme_dict = {iph:ii for ii, iph in enumerate(unique_phonemes)}
phoneme_data = np.zeros([len(audio), len(unique_phonemes), len(tfrs.times)])

# Loop through phonemes and populate our binary data with its timings
for iep, isentence in phonemes.groupby('number'):
    for _, (istt, istp, iph, _) in isentence.iterrows():
        mask = mne.utils._time_mask(tfrs.times, istt, istp)
        phoneme_data[iep, phoneme_dict[iph], mask] = 1

# phoneme_dict tells us which row is associated w/ each phoneme
# This inserts the phoneme names into their proper place
phonemes_rev = dict((jj, ii) for ii, jj in phoneme_dict.items())
phonemes_labels = [phonemes_rev[ii] for ii in np.sort(list(phoneme_dict.values()))]

# Turn phoneme data into an EpochsArray for saving
info = mne.create_info(phonemes_labels, sfreq_new, 'misc')
phoneme_data = mne.EpochsArray(phoneme_data, info, tmin=tfrs.times.min())

fig, ax = plt.subplots()
ax.pcolormesh(phoneme_data.times, range(len(unique_phonemes)),
              phoneme_data._data[10], cmap=plt.cm.Greys)
ax.set_ylabel('Phoneme ID (a.u.)')
ax.set_xlabel('Time (s)')
ax.set_title('Phoneme presence for one sentence')

phoneme_data.save(save_path + 'phonemes-epo.fif')

fig, axs = plt.subplots(3, 1, sharex=True)
ix = 2
# Plot the envelope
# Note that some values may be negative due to the filtering
axs[0].plot(envelope.times, envelope._data[ix][0])
axs[0].set_title('Speech Envelope')

# Plot the spectrogram
axs[1].pcolormesh(tfrs.times, frequencies, np.log(tfrs._data[ix]),
                  cmap=plt.cm.viridis, vmin=10, vmax=20)
axs[1].set_title('Spectrogram')

# Plot the phoneme labeling
axs[2].pcolormesh(tfrs.times, range(len(unique_phonemes)),
                  phoneme_data._data[ix], cmap=plt.cm.Greys)
axs[2].set_title('Phonemes')
axs[2].set_xlabel('Time (s)')
for ax in axs:
    ax.axis('tight')
plt.tight_layout()

phone_lengths = np.sort(phonemes['stop'] - phonemes['start'])[::-1]

# We'll take a fixed window after each phoneme's onset
# The length of this window will be the mean length of all phonemes
time_keep_phone = np.mean(phone_lengths)
n_ixs_keep = int(time_keep_phone * high_freq.info['sfreq'])

# Finally, we'll create an Epoched version of *each* phoneme for classification
epochs_ph = []
for ii, phns in phonemes.groupby('number'):
    for _, (phst, phstp, phn, _) in phns.iterrows():
        ix_stt = int(phst * high_freq.info['sfreq'])
        i_mask_time = np.zeros(high_freq._data.shape[-1], dtype=bool)
        i_mask_time[ix_stt:ix_stt + n_ixs_keep] = True
        epochs_ph.append((phn, high_freq._data[ii][..., i_mask_time]))

phones, epochs_ph = zip(*epochs_ph)
phones = np.hstack(phones)
phones_labels = [phoneme_dict[ii] for ii in phones]
epochs_ph = np.stack(epochs_ph)

# Create an MNE representation of these event onsets.
events_phones = np.vstack([np.arange(len(phones_labels)),
                           np.zeros_like(phones_labels),
                           phones_labels]).T
# Now turn into an MNE object
epochs_phones = mne.EpochsArray(epochs_ph, high_freq.info, events_phones, event_id=phoneme_dict, tmin=0)

epochs_phones

# It appears that some channels are responsive to this phoneme on average
_ = epochs_phones['aa'].average().plot()

epochs_phones.save(save_path + 'ecog_phonemes-epo.fif')

# For labeling axes, what's the step between frequencies
freq_step_log = np.log(frequencies)
freq_step_log = freq_step_log[1] - freq_step_log[0]

# Calculate labels for our X and Y axes
mps_times = fftshift(fftfreq(tfrs.times.shape[0],
                             1. / tfrs.info['sfreq']))
mps_freqs = fftshift(fftfreq(frequencies.shape[0], freq_step_log))

# Loop through each epoch of our spectrograms and calculate the MPS
mps_all = []
for spec in tfrs:
    mps = fft2(spec)
    mps_all.append(np.abs(fftshift(mps)))
mps_all = np.array(mps_all)

# We need to use inline plotting because this won't work interactively
get_ipython().magic('matplotlib inline')
fig = plot_gabors_2d()

# Plot a sample MPS and the mean across all
fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
for ax, mps_plt in zip(axs, [np.log(mps_all[0]), np.log(mps_all).mean(0)]):
    ax.pcolormesh(mps_times, mps_freqs, mps_plt, cmap='viridis')
    ax.contour(mps_times, mps_freqs, mps_plt,
               np.percentile(mps_plt, [80, 90, 95, 99]))
    _ = plt.setp(ax, xlim=[-10, 10], ylim=[0, 9])
axs[0].set_title('One MPS')
axs[1].set_title('Mean MPS')
axs[0].set_xlabel('Rate Modulation\n(cyc / second)')
axs[0].set_ylabel('Spectral Modulation\n(cyc / octave)')

# Loop through words and tidy up each one's timing
words = []
for ii, stim_name in enumerate(stimuli['stim_name']):
    stim_name = stim_name.replace('.wav', '')
    with open(raw_path + 'timit_word_info/text_info/{}.WRD'.format(stim_name, 'r')) as f:
        w_sentence = f.readlines()
        w_sentence = [jj.replace('\n', '').split(' ') for jj in w_sentence]
        w_sentence = pd.DataFrame(w_sentence)
        w_sentence['number'] = ii
        words.append(w_sentence)
words = pd.concat(words)
words.columns = ['start', 'stop', 'word', 'number']

words[['start', 'stop']] = words[['start', 'stop']].apply(pd.to_numeric)
words[['start', 'stop']] /= sfreq_timit

# Now create a categorical variable for each word and populate with 1s
unique_words = words['word'].unique()
word_dict = {iph:ii for ii, iph in enumerate(unique_words)}
word_data = np.zeros([len(audio), len(unique_words), len(envelope.times)])

for iep, isentence in words.groupby('number'):
    for _, (istt, istp, iph, _) in isentence.iterrows():
        mask = mne.utils._time_mask(envelope.times, istt, istp)
        word_data[iep, word_dict[iph], mask] = 1

fig, axs = plt.subplots(2, 1)
mask_time = mne.utils._time_mask(envelope.times, -.5, 3)
ix = 20
axs[0].plot(envelope.times[mask_time],
            envelope._data[ix][0][..., mask_time])
axs[0].set_title('Speech Envelope')

axs[1].pcolormesh(envelope.times[mask_time], range(len(unique_words)),
                  word_data[ix][..., mask_time], cmap=plt.cm.Greys)
axs[1].set_title('Word Presence')
axs[1].set_ylabel('Unique Word ID')
axs[1].set_xlabel('Time (s)')
for ax in axs:
    ax.axis('tight')
plt.tight_layout()

# First we'll create delayed versions of the spectrogram.
# This lets us do PCA across patterns in both time and frequency.
delays = np.arange(0, -.4, -.02)
mask_time = mne.utils._time_mask(tfrs.times, 0, 4)

# We'll use the log of the data so frequencies are closer in amplitude
delayed = delay_time_series(np.log(tfrs._data), delays, tfrs.info['sfreq'])
delayed = delayed[..., mask_time]
delayed = delayed.reshape(delayed.shape[0], -1, delayed.shape[-1])
delayed = np.hstack(delayed)

# We'll use scikit-learn to extract the principle components of these features
# Scaling the data makes sure that PCs aren't dominated by amplitude differences
decomp = PCA()
decomposed = decomp.fit_transform(scale(delayed.T))

fig, ax = plt.subplots()
ax.plot(decomp.explained_variance_ratio_[:50])
ax.set_title('Explained variance of PCs')
ax.set_xlabel('PC Number')
ax.set_ylabel('Ratio of explained variance')

# Reshape the component weights so it's shape (n_frequencies, n_delays)
ix_component = 0
plt_component = decomp.components_[ix_component].reshape(-1, len(delays))

fig, ax = plt.subplots(figsize=(6, 6))
ax.pcolormesh(delays, np.log10(frequencies), plt_component,
              cmap=plt.cm.viridis)
ax.set_yticks(np.log10(frequencies)[::10])
ax.set_yticklabels(np.round(frequencies[::10]))
ax.axis('tight')
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time Lag (s)')
ax.set_title('Feature coefficients for PC {}'.format(ix_component))
_ = plt.setp(ax.get_xticklabels(), rotation=45)
plt.tight_layout()

