import os
import sys
import time
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

plt.ion()
plt.rcParams["image.cmap"] = "gist_gray"

dpath  = "images"
fname  = "city_image.jpg"
infile = os.path.join(dpath, fname)
img    = nd.imread(infile)

nrow, ncol = img.shape[:2]
rind = np.arange(nrow * ncol).reshape(nrow, ncol) // ncol
cind = np.arange(nrow * ncol).reshape(nrow, ncol) % ncol

fig8, ax8 = plt.subplots(1, 2, num=8, figsize=(8, 3))
ax8[0].imshow(rind)
ax8[0].grid(0)
ax8[1].imshow(cind)
ax8[1].grid(0)

mask = np.dstack([(rind < 200).astype(np.uint8) for i in range(3)])

mask.shape

ysize = 3.
xsize = ysize*float(img.shape[1]) / float(img.shape[0])

fig9, ax9 = plt.subplots(num=9, figsize=[xsize, ysize])
fig9.subplots_adjust(0, 0, 1, 1)
ax9.axis('off')
im9 = ax9.imshow(img * mask)
fig9.canvas.draw()

rm, cm = 244, 302
dist = np.sqrt((rind - rm)**2 + (cind - cm)**2)

im9.set_data(dist)
im9.set_clim(0, 500)
fig9.canvas.draw()

dist.shape

mask = np.zeros(img.shape, dtype=np.uint8)

mask.shape

(dist <= 100).shape

mask[dist <= 100] = [1, 1, 1, 4]

mask[dist <= 100] = [1, 1, 1]

im9.set_data(255 * mask)
fig9.canvas.draw()

im9.set_data(img * mask)
fig9.canvas.draw()

xpos, ypos = fig9.ginput()[0]

print("xpos = {0}\nypos = {1}".format(xpos, ypos))

get_ipython().magic('pinfo fig9.ginput')

print fig9.ginput(3)

cpos, rpos = [int(round(i)) for i in fig9.ginput()[0]]

print("rpos = {0}\ncpos = {1}".format(rpos, cpos))

plt.close("all")



