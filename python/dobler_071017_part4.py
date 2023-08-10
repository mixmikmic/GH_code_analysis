import os
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

plt.ion() # set interactive matplotlib
plt.rcParams["image.cmap"] = "gist_gray"

# -- read the image and get attributes
dpath = "images"
fname = "city_image.jpg"
img   = nd.imread(os.path.join(dpath, fname))
nrow = img.shape[0]
ncol = img.shape[1]

# -- display the image
xs   = 10.
ys   = xs * float(nrow) / float(ncol)
fig0, ax0 = plt.subplots(num=0, figsize=(xs, ys))
fig0.subplots_adjust(0, 0, 1, 1)
ax0.axis("off")
im0 = ax0.imshow(img)
fig0.canvas.draw()

neg  = 255 - img
imgs = [img, neg]

neg_flag = [True] # a flag to determine the state of the displayed image

def toggle(event):
    """
    Toggle between two images (defined outside of this 
    function as a list named "imgs")
    """
    
    # -- if the "n" key is pressed
    if event.key == "n":

        # flip the display flag
        neg_flag[0] = ~neg_flag[0]
        
        # reset the data
        im0.set_data(imgs[neg_flag[0]])
        fig0.canvas.draw()
        
    return

dum = fig0.canvas.mpl_connect("key_press_event", toggle)

plt.close("all")



