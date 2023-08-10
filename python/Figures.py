# import modules
import numpy as np
import librosa
import music_gen_lib as mgl
from keras.utils import np_utils
import time
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set(context="paper", style="white", font_scale=2)
default_figure_size = (6, 4)
default_font_size = 22

# load example wavefile
example_audio_file = "blues.00000.au"
x, sr = librosa.core.load(example_audio_file)

plt.figure(figsize=default_figure_size)
plt.plot(np.arange(1, len(x)+1) * 1.0 / sr, x, linewidth=0.11)
plt.xlim(0, 30); plt.xlabel("time (seconds)", fontsize=18); plt.ylabel("magnitude"); plt.tight_layout()
# plt.savefig("example_waveform.png", dpi=300)
plt.show()

# convert waveform into spectrogram
plt.figure(figsize=default_figure_size)
plt.specgram(x, NFFT=512, Fs=sr, cmap="jet"); 
plt.xlim(0, 30); plt.xlabel("time (seconds)"); plt.ylabel("Frequency (Hz)"); plt.tight_layout()
# plt.savefig("example_spectrogram.png", dpi=300)
plt.show()

example_mel_spectrogram = mgl.mel_spectrogram(x, sr)
plt.figure(figsize=(default_figure_size[0]+1, default_figure_size[1]))
plt.imshow(np.log(example_mel_spectrogram+1), origin="lower", aspect="auto", cmap="jet")
plt.xticks(np.linspace(0, 2587, 7), np.arange(0, 31, 5))
plt.xlabel("time (seconds)"); plt.ylabel("mel scale"); plt.colorbar(); plt.tight_layout()
# plt.savefig("example_mel_spectrogram_colorbar.png", dpi=300)
plt.show()

plt.figure(figsize=(default_figure_size[0]+1, default_figure_size[1]))
plt.imshow(np.log(example_mel_spectrogram+1)[:, :258], origin="lower", aspect="auto", cmap="jet")
plt.xticks(np.linspace(0, 258, 7), np.arange(0, 3.1, 0.5))
plt.xlabel("time (seconds)"); plt.ylabel("mel scale"); 
plt.colorbar()
plt.tight_layout()
# plt.savefig("example_mel_spectrogram_segment_colorbar.png", dpi=300)
plt.show()

random_seed = 0
saved_model_name = "mgcnn_rs_" + str(random_seed) + ".h5"
# saved_model_name = "mgcnn_poisson_rs_" + str(random_seed) + ".h5"
MGCNN = mgl.Music_Genre_CNN(mgl.baseline_model)
MGCNN.load_model(saved_model_name)

