import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

get_ipython().magic('matplotlib inline')

inFile = r'..\data\Kevitsa_geology_noframe.png'
imRGB = plt.imread(inFile)
# plot
fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(imRGB)
plt.title('Original RGB image')

imRGB.shape

imRGB = imRGB[:,:,:3]

inFile = r'..\data\Windows_256_color_palette_RGB.csv'
win256 = np.loadtxt(inFile,delimiter=',')

win256[:5]

nrows,ncols,d = imRGB.shape
flat_array = np.reshape(imRGB, (nrows*ncols, 3))
flat_array[:5]

# import function
from sklearn.metrics import pairwise_distances_argmin
# run function, making sure the palette data is normalised to the 0-1 interval
indices = pairwise_distances_argmin(flat_array,win256/255)
# reshape the indices to the shape of the initial image
indexedImage  = indices.reshape((nrows,ncols))

fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(indexedImage,cmap='viridis')
plt.title('Quantization with nearest distance to win256')

new_cm = mcolors.LinearSegmentedColormap.from_list('win256', win256/255)
plt.register_cmap(cmap=new_cm)  # optional but useful to be able to call the colormap by its name.

fig,ax = plt.subplots(figsize=(6,6))
ax.imshow(indexedImage,cmap='win256',norm=mcolors.NoNorm())
plt.title('Quantization with nearest distance to win256')

outFile = r'..\data\Kevitsa_geology_indexed.npy'
np.save(outFile,indexedImage)

fig, ax = plt.subplots(figsize=(4,4))
ax.imshow(np.arange(256).reshape(16, 16),
          cmap = 'win256',
          interpolation="nearest", aspect="equal")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(False)
ax.set_title('win256: Windows 8-bit palette')

