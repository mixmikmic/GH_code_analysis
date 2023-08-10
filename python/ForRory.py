get_ipython().magic('matplotlib inline')

import numpy as np  # library for manipulating arrays
import pylab as pl  # library for making plots
import cv2          # OpenCV library

fakeimage = np.zeros((128,128))
fakeimage[31:96,31:96] = 10.0

pl.subplot(111)
pl.imshow(fakeimage)
pl.show()

fakeimage+=np.random.normal(0,0.02,size=(128,128))

pl.subplot(111)
pl.imshow(fakeimage)
pl.imsave("image.png",fakeimage)
pl.show()

img= cv2.imread('image.png',0)

# Use Otsu filtering to determine thresholds for finding the edge:
high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
low_thresh = 0.9*high_thresh

# find the edges using Canny method:
edges = cv2.Canny(img,low_thresh,high_thresh)

# plot the output:
pl.plot(1),pl.imshow(edges,cmap = 'gray')
pl.show()

n_real = 100 # number of realisations

for i in range(0,n_real):
    
    # same underlying pattern:
    fakeimage = np.zeros((128,128))
    fakeimage[31:96,31:96] = 10.0
    
    # add a new noise realisation:
    fakeimage+=np.random.normal(0,0.02,size=(128,128))
    
    # save it to an image:
    pl.imsave("image_"+str(i)+".png",fakeimage)

all_edges = np.zeros(edges.shape)
for i in range(0,n_real):

    img= cv2.imread('image_'+str(i)+'.png',0)

    # Use Otsu filtering to determine thresholds for finding the edge:
    high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.9*high_thresh

    # find the edges using Canny method:
    edges = cv2.Canny(img,low_thresh,high_thresh)
    
    all_edges+=edges

# plot the output:
pl.plot(1),pl.imshow(all_edges,cmap = 'gray')
pl.show()



