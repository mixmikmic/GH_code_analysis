from __future__ import print_function
import librosa
import librosa.display
import IPython.display
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
get_ipython().run_line_magic('matplotlib', 'inline')

y, sr = librosa.load('xc_audio_182221_967.mp3', duration=30, sr=22050) #load only 30 seconds
y2, sr2 = librosa.load('train_sample/bluethroat-luscinia-svecica/xc_audio_371231_902.mp3', duration=30, sr=22050) #load only 30 seconds
'''
xc_audio_182221_967.mp3
xc_audio_315862_802.mp3
xc_audio_364900_910.mp3
xc_audio_370479_904.mp3
xc_audio_371231_902.mp3
'''

# len(y2)

# Play it back!
IPython.display.Audio(data=y, rate=sr)


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

#y, sr = librosa.load('train_sample/bluethroat-luscinia-svecica/xc_audio_182221_967.mp3', duration=30, sr=22050) 
#y, sr = librosa.load('train_sample/bluethroat-luscinia-svecica/xc_audio_371231_902.mp3', duration=30, sr=22050) 
#y, sr = librosa.load('train_sample/bluethroat-luscinia-svecica/xc_audio_315862_802.mp3', duration=30, sr=22050) 
#y, sr = librosa.load('train_sample/bluethroat-luscinia-svecica/xc_audio_364900_910.mp3', duration=30, sr=22050) 
#y, sr = librosa.load('train_sample/bluethroat-luscinia-svecica/xc_audio_370479_904.mp3', duration=30, sr=22050)
#y, sr = librosa.load('train_sample/common_chaffich/xc_audio_123500_39.mp3', duration=30, sr=22050)
#y, sr = librosa.load('train_sample/common_chaffich/xc_audio_214660_933.mp3', duration=30, sr=22050)
#y, sr = librosa.load('train_sample/common_chaffich/xc_audio_243499_776.mp3', duration=30, sr=22050)
# y, sr = librosa.load('train_sample/common_chaffich/xc_audio_95130_488.mp3', duration=30, sr=22050)
# y, sr = librosa.load('train_sample/common_chaffich/xc_audio_96308_172.mp3', duration=30, sr=22050)
#y, sr = librosa.load('train_sample/house-wren-troglodytes-aedon/xc_audio_21722_902.mp3', duration=30, sr=22050)
#y, sr = librosa.load('train_sample/house-wren-troglodytes-aedon/xc_audio_132446_247.mp3', duration=30, sr=22050)
y, sr = librosa.load('train_sample/house-wren-troglodytes-aedon/xc_audio_317962_464.mp3', duration=30, sr=22050)

'''

train_sample/house-wren-troglodytes-aedon/xc_audio_338185_300.mp3
train_sample/house-wren-troglodytes-aedon/xc_audio_352451_571.mp3
'''

#default=> S=None, n_fft=2048, hop_length=512, **kwargs

S = librosa.stft(y)
M = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=4096, hop_length=1024,n_mels=128,fmax=11025)

# S2 = librosa.stft(y2)
# M2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_fft=4096, hop_length=1024, n_mels=128,fmax=11025)

plt.clf()
plt.figure(figsize=(4,4))
plt.subplot(2,2,1)
# librosa.display.specshow(librosa.logamplitude(M2,ref_power=np.max), y_axis='log', fmax=11025, \
                           # x_axis='time',cmap='gray')
librosa.display.specshow(librosa.logamplitude(M,ref_power=np.max),sr=sr, hop_length=1024,                         fmax=11025,cmap='gray')


# plt.colorbar(format='%+2.0f dB')
#plt.title('Mel spectrogram')
plt.tight_layout()
plt.savefig('test2_house_features.png', bbox_inches='tight',dpi=72)

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

