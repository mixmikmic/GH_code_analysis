import cv2
import glob
import numpy as np
import math
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

DICT_PATH = '../data/dict.npy'
NUM_SHOW_ENTRIES = 200

D = np.load(DICT_PATH)

fig, axes = plt.subplots(ncols=10, nrows=math.ceil(NUM_SHOW_ENTRIES/10), figsize=(10,NUM_SHOW_ENTRIES/10))
plt.axis('off')
for i, entry in enumerate(D[:, :NUM_SHOW_ENTRIES].T):
    ax=axes[i//10, i%10]
    ax.imshow(entry.reshape(8,8), cmap = plt.get_cmap('gray')) # , vmin = 0, vmax = 255
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_axis_off()
    
for i in range(NUM_SHOW_ENTRIES, 10*math.ceil(NUM_SHOW_ENTRIES/10)):
    ax=axes[i//10, i%10]
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_axis_off()
plt.savefig('dictionary.png', transparent=True)
plt.show()



