get_ipython().magic('matplotlib inline')
import sys
sys.path.insert(0,'..')
from IPython.display import HTML,Image,SVG,YouTubeVideo
from helpers import header

HTML(header())

def norm_hist(ima):
    hist,bins = np.histogram(ima.flatten(),range(256))  # histogram is computed on a 1D distribution --> flatten()
    return 1.*hist/np.sum(hist) # normalized histogram

def display_hist(ima):
    plt.figure(figsize=[10,5])
    if ima.ndim == 2:
        nh = norm_hist(ima)
    else:
        nh_r = norm_hist(ima[:,:,0])
        nh_g = norm_hist(ima[:,:,1])
        nh_b = norm_hist(ima[:,:,2])
    # display the results
    plt.subplot(1,2,1)
    plt.imshow(ima,cmap=plt.cm.gray)
    plt.subplot(1,2,2)
    if ima.ndim == 2:
        plt.plot(nh,label='hist.')
    else:
        plt.plot(nh_r,color='r',label='r')
        plt.plot(nh_g,color='g',label='g')
        plt.plot(nh_b,color='b',label='b')
    plt.legend()
    plt.xlabel('gray level');

from skimage.data import camera,coins
from skimage.filter import sobel
import matplotlib.pyplot as plt
import numpy as np

im = coins().astype(np.float)
plt.imshow(im,cmap=plt.cm.gray);

fsobel = sobel(im) 
norm = 255*fsobel/np.max(fsobel)

display_hist(norm)

import bokeh.plotting as bk
from helpers import bk_image,bk_image_hoover,bk_compare_image

bk.output_notebook()


borders = 255*(norm>50)
bk_image(borders)

import skimage.filters.rank as skr
from skimage.morphology import disk

fim = skr.median(im.astype(np.uint8),disk(7))
fsobel = sobel(fim) 
norm = 255*fsobel/np.max(fsobel)

plt.imshow(norm>30,cmap=plt.cm.gray)



