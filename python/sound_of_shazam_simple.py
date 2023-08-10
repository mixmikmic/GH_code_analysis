get_ipython().magic('pylab inline')
import IPython
import librosa as lr

# read audio
y, sr = lr.load('rick.wav', sr=44100, mono=True)

# stft
n = 4096
hop = n/2
Y = lr.stft(y, n_fft=n, hop_length=hop)

# crop the spectrogram
fmax = 8000.0
maxbin = int(fmax/sr*n)
Yc = Y[:maxbin,:]

# magnitudes
Ym = np.abs(Yc)

# normalize
Yn = Ym/np.max(Ym)

# plot sample
fig, ax = subplots(figsize=(12, 4))
ax.imshow(Yn, origin='lower', interpolation='nearest', aspect='auto', cmap='binary')

from scipy.ndimage.morphology import grey_dilation

# mask size
mask_size = (32,20)

# compute mask
mask = grey_dilation(Yn, size=mask_size)

# plot mask
fig, ax = subplots(figsize=(12, 4))
ax.imshow(mask, origin='lower', interpolation='nearest', aspect='auto', cmap='binary')

# peak detection trick: peak locations <=> image == mask
peaks = Yn*(Yn==mask)

# plot peaks
fig, ax = subplots(figsize=(12, 4))
ax.imshow(Yn==mask, origin='lower', interpolation='nearest', aspect='auto', cmap='binary')

Z = np.zeros(Y.shape, complex)
Z[:peaks.shape[0],:] = peaks

# istft the result...
z = lr.istft(Z, hop_length=hop)

# let's listen!
IPython.display.Audio(data=z, rate=sr)

