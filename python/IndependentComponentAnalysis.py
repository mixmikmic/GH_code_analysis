import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import IPython.display
import os
import librosa
from pydub import AudioSegment as audio
from sklearn.decomposition import FastICA, PCA

# obtain two audio files for comparison
arun = '/home/cc/Audio/Train/arun_alphabet.wav'
andrew = '/home/cc/Audio/Train/andrew_alphabet.wav'
arun_pydub = audio.from_wav(arun)
andrew_pydub = audio.from_wav(andrew)

# depending on which is longer, shorten to same length
if arun_pydub.duration_seconds > andrew_pydub.duration_seconds:
    arun_pydub = arun_pydub[0:int(andrew_pydub.duration_seconds)*1000]
    andrew_pydub = andrew_pydub[0:int(andrew_pydub.duration_seconds)*1000]
else:
    andrew_pydub = arun_pydub[0:int(arun_pydub.duration_seconds)*1000]
    arun_pydub = andrew_pydub[0:int(arun_pydub.duration_seconds)*1000]
print(arun_pydub.duration_seconds, andrew_pydub.duration_seconds)

# now that the audio files are the same length, save them then import using librosa
andrew_pydub.export('audio_samples/andrew.wav', format='wav')
arun_pydub.export('audio_samples/arun.wav', format='wav')
arun, fs1 = librosa.load('audio_samples/arun.wav')
andrew, fs2 = librosa.load('audio_samples/andrew.wav')

# combine the two into one numpy array, add noise
S = np.c_[arun, andrew]

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1], [0.5, 2], [1.5, 1.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=2)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# For comparison, compute PCA
pca = PCA(n_components=2)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# graph it

plt.figure()

models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.5, 1)
plt.show()

# separate ICA analysis into the two parts
ica1 = S_[:,0]
ica2 = S_[:,1]

IPython.display.Audio(ica1, rate=fs1)

IPython.display.Audio(ica2, rate=fs2)

# plot both original and then separated audio files
plt.figure()
plt.subplot(4,1,1)
librosa.display.waveplot(andrew, sr=fs2)
plt.title('Andrew Original Signal')
plt.subplot(4,1,2)
librosa.display.waveplot(ica2, sr=fs2)
plt.title('Andrew Recovered Signal')
plt.subplot(4,1,3)
librosa.display.waveplot(arun, sr=fs1)
plt.title('Arun Original Signal')
plt.subplot(4,1,4)
librosa.display.waveplot(ica1, sr=fs1)
plt.title('Arun Recovered Signal')

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.5, 1)
plt.show()

mitha = '/home/cc/mitha_alphabet.wav'
mitha_pydub = audio.from_wav(mitha)

mitha_pydub = mitha_pydub[:int(andrew_pydub.duration_seconds)*1000]
print(mitha_pydub.duration_seconds)

mitha_pydub.export('audio_samples/mitha.wav', 'wav')
mitha, fs3 = librosa.load('audio_samples/mitha.wav')

# combine the two into one numpy array, add noise
S = np.c_[arun, andrew, mitha]
S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# graph it

plt.figure()

models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.5, 1)
plt.show()

# separate into recovered audios
sep = []
sep.append(S_[:,0])
sep.append(S_[:,1])
sep.append(S_[:,2])

# resulting audio of the ICA separation
IPython.display.display(IPython.display.Audio(sep[0], rate=fs1))
IPython.display.display(IPython.display.Audio(sep[1], rate=fs2))
IPython.display.display(IPython.display.Audio(sep[2], rate=fs3))

# mixed audio for comparison. notice that they are completely mixed using the mixing array, as specified before
IPython.display.display(IPython.display.Audio(X[:,0], rate=fs1))
IPython.display.display(IPython.display.Audio(X[:,1], rate=fs2))
IPython.display.display(IPython.display.Audio(X[:,2], rate=fs3))



