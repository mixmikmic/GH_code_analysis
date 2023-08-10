get_ipython().magic('matplotlib inline')
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

pylab.rcParams['figure.figsize'] = 16, 8
pylab.rcParams['font.size'] = 16

fs, samples = wavfile.read('audio/brooklyn_street.wav')

dur = float(len(samples)) / fs
sample_len = len(samples)

print 'Sample rate: ', fs, 'Hz'
print 'Length in samples: ', sample_len
print 'Length in seconds: ', '%.3f' % dur
print 'Sample data type: ', samples.dtype

samples = samples / 32768.0

T = np.linspace(0, dur, num=sample_len)

plt.plot(T, samples)
plt.ylim([-1.1, 1.1])
plt.xlim([0, np.max(T)])
plt.title('Brooklyn street')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

ab_samples = np.abs(samples)

plt.plot(T, ab_samples)
plt.ylim([0, 1.1])
plt.xlim([0, np.max(T)])
plt.title('Brooklyn street absolute amplitude')
plt.xlabel('Time (s)')
plt.ylabel('Absolute amplitude')

from scipy.signal import chirp

dur = 3

chir_samp_len = fs * dur

chirp_win_size = 1024

T_chirp = np.linspace(0, dur, chir_samp_len)

chirp_samps = chirp(T_chirp, f0=20, t1=dur, f1=20000, method='linear')

plt.specgram(chirp_samps, NFFT = chirp_win_size, noverlap = chirp_win_size/2, Fs = fs, mode = 'magnitude', scale = 'dB');
plt.ylim([0, 20000])
plt.xlim([0, np.max(T_chirp)])
plt.set_cmap('gist_rainbow_r')
plt.title('Chirp/linear swept sine - spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid(b=True, which='both', color='0.6',linestyle='--')

window_size = 2048

plt.specgram(samples, NFFT = window_size, noverlap = window_size/2, Fs = fs, mode = 'magnitude', scale = 'dB');
plt.ylim([0, 17000])
plt.xlim([0, np.max(T)])
plt.set_cmap('gist_rainbow_r')
plt.title('Brooklyn street - spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid(b=True, which='both', color='0.6',linestyle='--')



