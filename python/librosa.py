get_ipython().magic('matplotlib inline')

import librosa
import matplotlib.pyplot as plt
import seaborn as sns

# Load the example clip
y, sr = librosa.load(librosa.util.example_audio_file(), duration=10)
print(type(y), y.shape, sr)
print('Duration: {}'.format(librosa.get_duration(y=y, sr=sr)))

# Plot the signal
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
librosa.display.waveplot(y, sr=sr)

mfcc = librosa.feature.mfcc(y=y, sr=sr)
print(type(mfcc), mfcc.shape)

plt.figure(figsize=(10, 6))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()



