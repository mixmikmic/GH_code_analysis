import audio
audio.example()

import numpy as np
import specplotting
import matplotlib.pyplot as plt
d = np.load("./scratch/kaldispec.npy")
spec = d.tolist()["gasstation"]
specplotting.plot_spec(spec, sample_rate=16000, title="Kaldi Spectrogram")
plt.show()

