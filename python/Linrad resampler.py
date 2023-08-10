get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import scipy.io.wavfile as wavfile

# Larger figure size
fig_size = [14, 8]
plt.rcParams['figure.figsize'] = fig_size

samp_rate = 48000
pulse_on = samp_rate//100
pulse_off = samp_rate//10 - pulse_on
duration = 600 # seconds
n_pulses = duration * 10
amplitude = 0.01

pulse = np.array([1]*pulse_on + [0]*pulse_off, dtype='float32')
i = amplitude*np.tile(pulse, n_pulses)

iq = np.zeros((len(i),2), dtype='float32')
iq[:,0] = i

wavfile.write('/home/daniel/pulses.wav', rate=samp_rate, data=iq)
del iq
del i

def plot_audio(file):
    audio = np.memmap(file, offset=0x28, dtype=np.int16)
    #audio = np.abs(hilbert(audio)) # improves display, but takes computation time
    pulse_len = 4800
    lines = len(audio)//pulse_len
    plt.imshow(np.abs(audio)[:lines*pulse_len].reshape(lines, pulse_len), cmap='viridis', vmin = 0, vmax=1e4)
    del audio

plot_audio('/tmp/output-file-input.wav')

plot_audio('/tmp/output-alsa-ntpd-off.wav')

plot_audio('/tmp/output-alsa-ntpd-on.wav')

