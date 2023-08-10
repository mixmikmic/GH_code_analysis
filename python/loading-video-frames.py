# We also need Python's plotting library, matplotlib.
# change the following to %matplotlib notebook for interactive plotting
get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'  # Set grayscale images as default.

import trackpy as tp
import pims

v = pims.ImageSequence('../sample_data/bulk_water/*.png')

v[0]

plt.imshow(v[0]);

def gray(image):
    return image[:, :, 0]

v = pims.ImageSequence('../sample_data/bulk_water/*.png', process_func=gray)
plt.figure()
plt.imshow(v[0]);

for frame in v[2:5]:
    # Do something with each frame.
    pass

