from __future__ import print_function

import librosa
import librosa.display
import IPython.display
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the example track
# y, sr = librosa.load(librosa.util.example_audio_file())
# y, sr = librosa.load('output_test2.mp3')
y, sr = librosa.load('xc_audio_182221_967.mp3', duration=30)
#y, sr = librosa.load('voice_test.mp3')

#librosa.load?

# Play it back!
IPython.display.Audio(data=y, rate=sr)

#spectral centroid
centroid_y = librosa.feature.spectral_centroid(y=y, sr=sr)
centroid_y = centroid_y.T
x = range(0, centroid_y.shape[0], 1)
plt.plot(x, centroid_y)

centroid_y.mean()
plt.hlines(centroid_y.mean(),0,centroid_y.shape[0] ,color='r',alpha=0.5)
plt.show()

# How about separating harmonic and percussive components?
y_h, y_p = librosa.effects.hpss(y)

# Play the harmonic component
IPython.display.Audio(data=y_h, rate=sr)

# Play the percussive component
IPython.display.Audio(data=y_p, rate=sr)

# Pitch shifting?  Let's gear-shift by a major third (4 semitones)
y_shift = librosa.effects.pitch_shift(y, sr, 7)

IPython.display.Audio(data=y_shift, rate=sr)

# Or time-stretching?  Let's slow it down
y_slow = librosa.effects.time_stretch(y, 0.5)

IPython.display.Audio(data=y_slow, rate=sr)

frequencies, D = librosa.ifgram(y, sr=sr)
S, phase = librosa.magphase(D)
components, activations = librosa.decompose.decompose(S, n_components=4, sort=True)
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
librosa.display.specshow(librosa.amplitude_to_db(components, ref=np.max), y_axis='log')
plt.xlabel('Component')
plt.ylabel('Frequency')
plt.title('Components')

plt.subplot(1,2,2)
librosa.display.specshow(activations, x_axis='time')
plt.xlabel('Time')
plt.ylabel('Component')
plt.title('Activations')

plt.tight_layout()

# How about something more advanced?  Let's decompose a spectrogram with NMF, and then resynthesize an individual component
# D = librosa.stft(y)
C = librosa.cqt(y, sr)

# Separate the magnitude and phase
S, phase = librosa.magphase(C)

# Decompose by nmf
components, activations = librosa.decompose.decompose(S, n_components=8, sort=True)

# Visualize the components and activations, just for fun

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
librosa.display.specshow(librosa.amplitude_to_db(components, ref=np.max), y_axis='log')
plt.xlabel('Component')
plt.ylabel('Frequency')
plt.title('Components')

plt.subplot(1,2,2)
librosa.display.specshow(activations, x_axis='time')
plt.xlabel('Time')
plt.ylabel('Component')
plt.title('Activations')

plt.tight_layout()

D = librosa.stft(y)

# Separate the magnitude and phase
S, phase = librosa.magphase(D)

# Decompose by nmf
components, activations = librosa.decompose.decompose(S, n_components=8, sort=True)

# Visualize the components and activations, just for fun

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
librosa.display.specshow(librosa.amplitude_to_db(components, ref=np.max))

# plt.xlabel('Component')
# plt.ylabel('Frequency')
# plt.title('Components')

# plt.subplot(1,2,2)
# librosa.display.specshow(activations, x_axis='time')
# plt.xlabel('Time')
# plt.ylabel('Component')
# plt.title('Activations')

# plt.tight_layout()

print(components.shape, activations.shape)

#onsets and beats
#onset envelope strength:
onset_envelope = librosa.onset.onset_strength(y,sr)

#onset (new note) event detected:
onsets = librosa.onset.onset_detect(onset_envelope=onset_envelope)

# 
plt.figure(figsize=(16,4))

plt.subplot(2,1,1)
plt.plot(onset_envelope, label='Onset strength')
plt.vlines(onsets, 0, onset_envelope.max(), color='r', alpha=0.25, label = 'new note')
plt.xticks([]), plt.yticks([])
plt.legend()
plt.axis('tight')

plt.subplot(2,1,2)
librosa.display.waveplot(y_h, sr);

# full spectrum --Visualize the components and activations, just for fun

plt.figure(figsize=(16,4))

plt.subplot(2,1,1)
plt.plot(onset_envelope, label='Onset strength')
plt.vlines(onsets, 0, onset_envelope.max(), color='r', alpha=0.25, label = 'new note')
plt.xticks([]), plt.yticks([])
plt.legend()
plt.axis('tight')

plt.subplot(2,1,2)
librosa.display.waveplot(y, sr);

import mir_eval
tempo, beats = librosa.beat.beat_track(onset_envelope=onset_envelope)
beat_times = librosa.frames_to_time(onsets)
y_click = mir_eval.sonify.clicks(beat_times, sr, length=len(y))
IPython.display.Audio(data=y + y_click, rate = sr)

# D = librosa.stft(y)

# # Separate the magnitude and phase
# S, phase = librosa.magphase(D)

# # Decompose by nmf
# components, activations = librosa.decompose.decompose(S, n_components=8, sort=True)
plt.clf()
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
plt.xlabel('Time')
plt.ylabel('Pitch Class')
plt.title('real')

# plt.colorbar();
plt.subplot(2,2,2)
chroma = librosa.feature.chroma_stft(y=y_k, sr=sr)
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
plt.xlabel('Time')
plt.ylabel('Pitch class')
plt.title('reconstructed')

# plt.colorbar();
plt.subplot(2,2,3)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
plt.xlabel('Time')
plt.ylabel('Pitch class')
plt.title('real cqt')

# plt.colorbar();
plt.subplot(2,2,4)
chroma = librosa.feature.chroma_cqt(y=y_k, sr=sr)
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
plt.xlabel('Time')
plt.ylabel('Pitch class')
plt.title('reconstructed cqt')

#lets try to save something
plt.clf()
plt.figure(figsize=(16,4))

#create pitch pic with cqt and the  'real' data not reconstructed
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
librosa.display.specshow(chroma)
plt.savefig('test1_features.png', bbox_inches='tight')

# Mel spectra, MFC and Tonnetz
plt.figure(figsize=(16,4))

# M = librosa.feature.melspectrogram(y=y, sr=sr)
# librosa.display.specshow(M)
# plt.colorbar();
# MFCC = librosa.feature.mfcc(y=y, sr=sr)
# librosa.display.specshow(MFCC)
# plt.colorbar();
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
librosa.display.specshow(tonnetz)
plt.colorbar();

# Play back the reconstruction
# Reconstruct a spectrogram by the outer product of component k and its activation
D_k = components.dot(activations)

# invert the stft after putting the phase back in
y_k = librosa.istft(D_k * phase)

# And playback
print('Full reconstruction')

IPython.display.Audio(data=y_k, rate=sr)

# Resynthesize.  How about we isolate just first (lowest) component?
k = 0

# Reconstruct a spectrogram by the outer product of component k and its activation
D_k = np.multiply.outer(components[:, k], activations[k])

# invert the stft after putting the phase back in
y_k = librosa.istft(D_k * phase)

# And playback
print('Component #{}'.format(k))

IPython.display.Audio(data=y_k, rate=sr)

#y the numpy array of the original data input
#input spectrograms
spect_stft = np.abs(librosa.stft(y))
melspec = librosa.feature.melspectrogram(y=y, sr=sr)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

#reconstruction spectrographs
spect_stft_k = np.abs(librosa.stft(y_k))
melspec_k = librosa.feature.melspectrogram(y=y_k, sr=sr)
chroma_k = librosa.feature.chroma_cqt(y=y_k, sr=sr)
tonnetz_k = librosa.feature.tonnetz(y=y_k, sr=sr)
# print(spect_stft_k)

#load 2 files
#y, sr = librosa.load('train_sample/bluethroat-luscinia-svecica/xc_audio_315862_802.mp3')
#y, sr = librosa.load('train_sample/bluethroat-luscinia-svecica/xc_audio_182221_967.mp3')
##y, sr = librosa.load('train_sample/bluethroat-luscinia-svecica/xc_audio_364900_910.mp3')
#y, sr = librosa.load('train_sample/bluethroat-luscinia-svecica/xc_audio_370479_904.mp3')
#y, sr = librosa.load('voice_test.mp3')

y, sr = librosa.load('train_sample/bluethroat-luscinia-svecica/xc_audio_371231_902.mp3', duration=30)

y2, sr2 = librosa.load('train_sample/bluethroat-luscinia-svecica/xc_audio_182221_967.mp3', duration=30)

#D = np.abs(librosa.stft(y))**2 #why is this squared?
# S = librosa.feature.melspectrogram(S=D)

# Passing through arguments to the Mel filters
M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
S = librosa.stft(y)

M2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128,fmax=8000)
S2 = librosa.stft(y2)

plt.clf()
plt.figure(figsize=(16,20))
plt.subplot(4,1,1)
librosa.display.specshow(librosa.logamplitude(M,ref_power=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

plt.subplot(4,1,2)
librosa.display.specshow(librosa.logamplitude(M2,ref_power=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel2 spectrogram')
plt.tight_layout()

plt.subplot(4,1,3)
librosa.display.specshow(librosa.logamplitude(S,ref_power=np.max), y_axis='log', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('STFT spectrogram')
plt.tight_layout()

plt.subplot(4,1,4)
librosa.display.specshow(librosa.logamplitude(S2,ref_power=np.max), y_axis='log', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('STFT2 spectrogram')
plt.tight_layout()

get_ipython().run_line_magic('pinfo', 'librosa.display.specshow')

# Resynthesize.  How about we isolate a middle-frequency component?
k = len(activations) // 2

# Reconstruct a spectrogram by the outer product of component k and its activation
D_k = np.multiply.outer(components[:, k], activations[k])

# invert the stft after putting the phase back in
y_k = librosa.istft(D_k * phase)

# And playback
print('Component #{}'.format(k))

IPython.display.Audio(data=y_k, rate=sr)

# Resynthesize.  How about we isolate just last (highest) component?
k = -1

# Reconstruct a spectrogram by the outer product of component k and its activation
D_k = np.multiply.outer(components[:, k], activations[k])

# invert the stft after putting the phase back in
y_k = librosa.istft(D_k * phase)

# And playback
print('Component #{}'.format(k))

IPython.display.Audio(data=y_k, rate=sr)



