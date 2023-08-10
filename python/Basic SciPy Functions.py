from scipy.misc import imread, imsave, imresize

# Read an image into a numpy array
img = imread('bajrangbali')
print(img.dtype, img.shape)

# we can tint the image by by scaling each of the color channels by a different scalar constant.
# the image has shape (1200, 1920, 3); we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that it leaves the red channel unchanged and multiplies the green and blue channel by
# 0.95 and 0.9 respectively
img_tinted = img*[1, 0.95, 0.9]

# resize the tinted image to be 300 by 300 pixels
img_tinted = imresize(img_tinted, (300,300))

# write the tinted image back to disc
imsave('bajrangbali_tinted.jpg', img_tinted)

import numpy as np
from scipy.spatial.distance import pdist, squareform

# create an array where each row is a point in 2D space:
x = np.array([[0,1], [1,0], [2,0]])
print(x)

# compute euclidean distance between all rows of x.
d = squareform(pdist(x, 'euclidean'))
print(d)

