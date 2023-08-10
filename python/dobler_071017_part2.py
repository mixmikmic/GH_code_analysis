import os
import sys
import time
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

plt.ion() # set interactive matplotlib
plt.rcParams["image.cmap"] = "gist_gray"

dpath = 'images'
fname = 'ml.jpg'
infile = os.path.join(dpath, fname)
img_ml = nd.imread(infile)

print("file name is \"{0}\"".format(infile))

print("shape : {0}\ntype  : {1}".format(img_ml.shape, img_ml.dtype))

ysize = 6.
xsize = ysize * float(img_ml.shape[1]) / float(img_ml.shape[0])

fig5, ax5 = plt.subplots(num=5, figsize=[xsize, ysize])
fig5.subplots_adjust(0, 0, 1, 1)
ax5.axis("off")
im5 = ax5.imshow(img_ml)
fig5.canvas.draw()

im5.set_data(1.0 * img_ml)
fig5.canvas.draw()

fig6, ax6 = plt.subplots(1, 3, num=6, figsize=[3 * xsize, ysize])
fig6.subplots_adjust(0, 0, 1, 1, 0, 0)
[i.axis("off") for i in ax6]
im6a = ax6[0].imshow(img_ml)
im6b = ax6[1].imshow(0.25 * img_ml)
im6c = ax6[2].imshow((0.25 * img_ml).astype(np.uint8))
fig6.canvas.draw()

img_ml_L = img_ml.mean(2) # convert to gray scale by taking the mean across the color axis
print("img_ml_L\n  shape : {0}\n  type  : {1}"       .format(img_ml_L.shape, img_ml_L.dtype))

fig7, ax7 = plt.subplots(num=7, figsize=[xsize, ysize])
fig7.subplots_adjust(0, 0, 1, 1)
ax7.axis("off")
im7 = ax7.imshow(img_ml_L)
fig7.canvas.draw()

plt.close("all")



