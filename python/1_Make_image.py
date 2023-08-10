import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cmap = 'viridis'    # Perceptual
cmap = 'spectral'   # Classic rainbow
cmap = 'seismic'    # Classic diverging
cmap = 'Accent'     # Needs coolinearity constraint
cmap = 'Dark2'      # Needs coolinearity constraint
cmap = 'Paired'     # Needs coolinearity constraint, ultimate test!
cmap = 'gist_ncar'  # Works with new cool-point start location
cmap = 'Pastel1'    # Amazing that it works for start point
cmap = 'Set2'       # Difficult

cmap = 'RdBu'

from scipy import signal

nx, ny = 100, 100
z = np.random.rand(nx, ny)

sizex, sizey = 30, 30
x, y = np.mgrid[-sizex:sizex+1, -sizey:sizey+1]
g = np.exp(-0.333*(x**2/float(sizex)+y**2/float(sizey)))
f = g/g.sum()

z = signal.convolve(z, f, mode='valid')
z = (z - z.min())/(z.max() - z.min())

# Interpolation introduces new colours and makes it harder to recover the data.
plt.imshow(z, cmap=cmap)

# Prevent interpolation for the 'pure' experience.
#plt.imshow(z, cmap="spectral", interpolation='none')

plt.axis('off')
plt.savefig('data/test.png', bbox_inches='tight')
plt.show()

volume = np.load('data/F3_volume_3x3_16bit.npy')

# Choose a section and transpose it.
x = volume[20].T

# Clip the display at the 99.5% point.
vm = np.percentile(x, 99.5)

# Make figure
plt.figure(figsize=(14, 8), frameon=False)
plt.axis('off')

# Again: interpolation introduces new colours.
plt.imshow(x, cmap=cmap, interpolation='none', aspect='auto', vmin=-vm, vmax=vm)
plt.savefig('data/test.png', bbox_inches='tight')
plt.show()



