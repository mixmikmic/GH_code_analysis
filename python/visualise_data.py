DATASET_URL = "/Volumes/CB_RESEARCH/nasa/textures_v2_brown500_with_valid.h5"

# put your url here
#DATASET_URL = ""

import h5py
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

h5 = h5py.File(DATASET_URL, "r")

h5.keys()

rnd_idx = np.random.randint(0, h5['xt'].shape[0])
plt.subplot(1,2,1)
plt.imshow(h5['xt'][rnd_idx][:,:,0],cmap="gray")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(h5['yt'][rnd_idx],cmap="gray")
plt.axis('off')



