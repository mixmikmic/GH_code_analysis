import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

A = plt.imread("https://github.com/CambridgeEngineering/PartIA-Computing-Examples-Papers/raw/master/images/southwing.png")
print(type(A))

plt.imshow(A, cmap='gray');
print("Image array shape (pixels): {}".format(A.shape))

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import urllib

from IPython.display import Audio
get_ipython().magic('matplotlib inline')

# Sampling frequency
fs = 44100

# Time interval (seconds)
T = 1.5    

# Time points (0 to T, with T*fs points)
t = np.linspace(0, T, int(T*fs), endpoint=False)

# Signal frequencies
omega0, omega1  = 2*np.pi*240, 2*np.pi*440

# Create signal
x = np.sin(omega0*t) + np.sin(omega1*t)

# Plot signal over first 0.05 s
n = int(0.05*fs)
plt.plot(t[:n], x[:n])
plt.xlabel('time (seconds)')
plt.ylabel('$x(t)$');

Audio(x, rate=fs)

# Perform a Fourier transform of the signal (signal is real, so we can use the 'real' version)
xf = np.fft.rfft(x)

# Create frequency axis
freq = np.linspace(0.0, fs/2, len(xf))

# Plot Fourier coefficients against frequency. Fourier coefficients are complex, so
# we take the modulus.
plt.plot(freq, np.abs(xf))
plt.xlabel('frequency (Hz)')
plt.ylabel('$\hat{x}$');

plt.plot(freq, np.abs(xf))
plt.xlabel('frequency (Hz)')
plt.ylabel('$\hat{x}$')
plt.xlim(200, 500);

# Copy transformed problem
xf_filtered = xf.copy()

# Cut off requencies below 250 Hz
cutoff_freq = 250
n_cut = int(2*cutoff_freq*len(xf_filtered)/fs)
xf_filtered[:n_cut] = 0.0

plt.plot(freq, np.abs(xf_filtered))
plt.xlabel('frequency (Hz)')
plt.ylabel('$\hat{x}$')
plt.xlim(200, 500);

# Perform inverse transfiorm
x_filtered = np.fft.irfft(xf_filtered)

# Plot signal over first 0.05 s
n = int(0.05*fs)
plt.plot(t[:n], x_filtered[:n])
plt.xlabel('time (seconds)')
plt.ylabel('$x(t)$')

Audio(x_filtered, rate=fs)

url = "https://github.com/CambridgeEngineering/PartIA-Computing-Examples-Papers/raw/master/sound/piano1.wav"
Audio(url)

# Fetch sound file
local_filename, headers = urllib.request.urlretrieve(url)

# Read frequency and data array for sound track
fs, x = scipy.io.wavfile.read(local_filename) 

# If we have a stero track (left and right channels), take just the first channel
if len(x.shape) > 1:
    x = x[:, 0]

# Check that it plays
Audio(x, rate=fs)

# Time points (0 to T, with T*fs points)
t = np.linspace(0, len(x)/fs, len(x), endpoint=False)

# Plot signal
plt.plot(t, x)
plt.xlabel('time (seconds)')
plt.ylabel('signal');

# Perform discrete Fourier transform (real signal)
xf = np.fft.rfft(x)

# Create frequency axis for plotting
freq = np.linspace(0.0, fs/2, len(xf))

plt.semilogy(freq, np.abs(xf))
plt.xlabel('frequency (Hz)')
plt.ylabel('$\hat{x}$');

# Create copy og transformed signal
xf_filtered = xf.copy()

# Cut-off frequencies (Hz)
cutoff_freq_low = 1200
cutoff_freq_high = 1500

# Cut-off indices in transform array
n_cut_low = int(2*cutoff_freq_low*len(xf_filtered)/fs)
n_cut_high = int(2*cutoff_freq_high*len(xf_filtered)/fs)

# Remove low and high frequencies
xf_filtered[:n_cut_low] = 0.0
xf_filtered[n_cut_high:] = 0.0

# Plot filtered transform 
plt.semilogy(freq, np.abs(xf_filtered))
plt.xlabel('frequency (Hz)')
plt.ylabel('$\hat{x}$');

# Perform inverse transform on filtered signal
x_filtered = np.fft.irfft(xf_filtered)

# Plot signal
plt.plot(t, x_filtered)
plt.xlabel('Time (seconds)')
plt.ylabel('signal');

Audio(x_filtered, rate=fs)

