# import statements
import os
import pydub
import librosa
import tflearn
import tensorflow as tf
import librosa.display
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
get_ipython().magic('matplotlib inline')

# define the Two-Speaker network that contains 201 files from two speakers
Two_Speaker = '/home/cc/Data/Two-Speaker/'

# the log spectral magnitude is used as the input feature, will use librosa to find these
# as an example, will grab one audio file and display its log spectral magnitude
single_file = Two_Speaker + 'Speaker0_0.wav'
y, fs = librosa.load(single_file)

S = librosa.feature.melspectrogram(y=y, sr=fs, n_fft=32, hop_length=8)

librosa.display.specshow(S, sr=fs, y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.show()

y_stft = librosa.stft(y=y, n_fft=32, hop_length=8)

y_stft.shape

# the log spectral magnitude is used as the input feature, will use librosa to find these
# as an example, will grab one audio file and display its log spectral magnitude
single_file = '/home/cc/Data/Segment-Four/Speaker1_32.wav'
y, fs = librosa.load(single_file)
y_stft1 = librosa.stft(y=y, n_fft=32, hop_length=8)

y_stft1.shape

example = pydub.AudioSegment.from_wav(single_file)

a = example.frame_count()

a

frame_ms = a/1000

frame_ms

ms_per_frame = 100 / frame_ms

ms_per_frame

seg = example[:ms_per_frame*4]

seg.frame_count()

# grab all audio files in folder
audio_files = []
for f in os.listdir(Two_Speaker):
    audio_files.append(pydub.AudioSegment.from_wav(Two_Speaker + f))

# segment the data into 100-frame length
segments = []
for audio in audio_files:
    frame_ms = audio.frame_count() / 1000
    ms_per_frame = 100 / frame_ms
    segments

