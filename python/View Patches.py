import cv2
import glob
import numpy as np
import math
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

PATCH_PATH = '../data/patches/'
NUM_SHOW_PATCHES = 50

patches = np.random.choice(np.array(glob.glob(PATCH_PATH + '*.npy')), NUM_SHOW_PATCHES)

fig, axes = plt.subplots(ncols=3, nrows=math.ceil(NUM_SHOW_PATCHES/3), figsize=(10,NUM_SHOW_PATCHES))
plt.axis('off')
for i, patch in enumerate(patches):
    ax=axes[i//3, i%3]
    im = np.load(patch)
    ax.imshow(im) # , vmin = 0, vmax = 255
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_axis_off()
    
for i in range(NUM_SHOW_PATCHES, 3*math.ceil(NUM_SHOW_PATCHES/3)):
    ax=axes[i//3, i%3]
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_axis_off()

plt.savefig('randomPatches.png', transparent=True)



