import os
import sys
import time
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

plt.ion() # set interactive mpl
plt.rcParams["image.cmap"] = "gist_gray"

# -- 2. Load it into python using scipy.ndimage
img = nd.imread('images/city_image.jpg')

# -- 3. Display the full image
nrow, ncol = img.shape[:2]
xsize = 10.
ysize = xsize * float(nrow) / float(ncol)

fig0, ax0 = plt.subplots(num=0, figsize=[xsize, ysize])
fig0.subplots_adjust(0, 0, 1, 1)
ax0.axis("off")
im0 = ax0.imshow(img)
fig0.canvas.draw()

# -- 4. Display only the upper left corner
fig1, ax1 = plt.subplots(num=1, figsize=[xsize, ysize])
fig1.subplots_adjust(0, 0, 1, 1)
ax1.axis("off")
im1 = ax1.imshow(img[:nrow//2, :ncol//2])
fig1.canvas.draw()

# -- 5. Display only the lower right corner
fig2, ax2 = plt.subplots(num=2, figsize=[xsize, ysize])
fig2.subplots_adjust(0, 0, 1, 1)
ax2.axis("off")
im2 = ax2.imshow(img[nrow//2:, ncol//2:])
fig2.canvas.draw()

# -- 6. Display only the central half
fig3, ax3 = plt.subplots(num=3, figsize=[xsize, ysize])
fig3.subplots_adjust(0, 0, 1, 1)
ax3.axis("off")
im3 = ax3.imshow(img[nrow//4:3*nrow//4, ncol//4:3*ncol//4])
fig3.canvas.draw()

# -- 7. Display the negative of the full image
fig4, ax4 = plt.subplots(num=4, figsize=[xsize, ysize])
fig4.subplots_adjust(0, 0, 1, 1)
ax4.axis("off")
im4 = ax4.imshow(255 - img)
fig4.canvas.draw()

# -- 8. Reset the right half of the image as the negative of itself
img[:, ncol//2:] = 255 - img[:,ncol/2:]

fig5, ax5 = plt.subplots(num=5, figsize=[xsize, ysize])
fig5.subplots_adjust(0, 0, 1, 1)
ax5.axis("off")
im5 = ax5.imshow(img)
fig5.canvas.draw()

# -- 9. Plot a step function with a transition at ncol/2 and height nrow
xx = np.arange(ncol)
yy = nrow * (xx > ncol//2)

fig6, ax6 = plt.subplots(num=6, figsize=[xsize, ysize])
ax6.plot(xx, yy, color="red", lw=4)
fig6.canvas.draw()

# -- 10. Overshow the result of step 8
im6 = ax6.imshow(img)
fig6.canvas.draw()

plt.close("all")

plt.rcParams["backend"]



