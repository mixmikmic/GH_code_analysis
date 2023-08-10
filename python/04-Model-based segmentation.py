get_ipython().magic('matplotlib inline')
import sys
sys.path.insert(0,'..')
from IPython.display import HTML,Image,SVG,YouTubeVideo
from helpers import header

HTML(header())

Image('http://homepages.ulb.ac.be/~odebeir/data/hough3.png')

Image('http://homepages.ulb.ac.be/~odebeir/data/hough1.png')

Image('http://homepages.ulb.ac.be/~odebeir/data/hough4.png')

Image('http://homepages.ulb.ac.be/~odebeir/data/hough5.png')

#example from http://scikit-image.org/docs/dev/auto_examples/plot_line_hough_transform.html
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny

import numpy as np
import matplotlib.pyplot as plt

# Construct test image

image = np.zeros((100, 100))

# Classic straight-line Hough transform

idx = np.arange(25, 75)
image[idx[::-1], idx] = 255
image[idx, idx] = 255

h, theta, d = hough_line(image)

fig, ax = plt.subplots(1, 3, figsize=(8, 4))

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Input image')
ax[0].axis('image')

ax[1].imshow(np.log(1 + h),
           extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
                   d[-1], d[0]],
           cmap=plt.cm.gray, aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(image, cmap=plt.cm.gray)
rows, cols = image.shape
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
    ax[2].plot((0, cols), (y0, y1), '-r')
ax[2].axis((0, cols, rows, 0))
ax[2].set_title('Detected lines')
ax[2].axis('image');

from skimage.io import imread

rgb = imread('https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Garfield_Building_Detroit.jpg/320px-Garfield_Building_Detroit.jpg')
g = rgb[:,:,0]
ima = canny(g)

h, theta, d = hough_line(ima)

peaks = zip(*hough_line_peaks(h, theta, d))

plt.figure(figsize=[10,5])
plt.subplot(1,2,1)
plt.imshow(g,cmap=plt.cm.gray)
plt.subplot(1,2,2)
plt.imshow(ima,cmap=plt.cm.gray)

plt.figure(figsize=[18,8])
plt.imshow(np.log(1 + h),extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
                   d[-1], d[0]])
plt.colorbar()
for _, angle, dist in peaks:
    plt.plot(np.rad2deg(-angle),dist,'o')
    
plt.figure(figsize=[10,5])
plt.imshow(g, cmap=plt.cm.gray)
rows, cols = g.shape
for _, angle, dist in peaks:
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
    plt.plot((0, cols), (y0, y1), '-')
plt.axis((0, cols, rows, 0))
#plt.set_title('Detected lines')
#plt.axis('image');

Image('http://homepages.ulb.ac.be/~odebeir/data/hough_ex.png')



