from skimage import io
image = io.imread('https://upload.wikimedia.org/wikipedia/en/2/24/Lenna.png')

from skimage import color
imgray = color.rgb2gray(image)

import matplotlib.pyplot as plt

plt.imshow(image)
plt.show()

plt.imshow(imgray, cmap='gray')
plt.show()

import numpy as np
imbw = np.where(imgray > np.mean(imgray),1.0,0.0)   #Taking the average of the the grays and splitting them to get binary values 

plt.imshow(imbw, cmap = 'gray')
plt.show()

