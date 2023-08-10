get_ipython().magic('matplotlib inline')
import os
import ClearMap.IO as io
import ClearMap.Settings as settings
filename = os.path.join(settings.ClearMapPath, 'data/clarity0012.tif')

import ClearMap.Visualization.Plot as clrplt

# data = io.readData(filename);
# clrplt.plotTiling(data);
#'/root/plotter.py'

#import subprocess
#with open("/root/output.png", "w+") as output:
#    subprocess.call(["python", "/root/plotter.py"], stdout=output);

data = io.readData(filename);
#clrplt.plotTiling(data);
#clrplt.plotTiling(data, inverse = True);

#Tried to hack the code by manually editing the backend plot generation to save images. Didn't work'

# from PIL import Image

# img = Image.open('/root/output.png')
# img.show() */

clrplt.plotTiling(data, inverse = False);

clrplt.plotTiling(data, inverse = True);

import ClearMap.ImageProcessing.BackgroundRemoval as bgr
dataBGR = bgr.removeBackground(data.astype('float'), verbose = True);
clrplt.plotTiling(dataBGR, inverse = True);

from ClearMap.ImageProcessing.Filter.DoGFilter import filterDoG
dataDoG = filterDoG(dataBGR, verbose = True);
clrplt.plotTiling(dataDoG, inverse = True, z = (10,16));

from ClearMap.ImageProcessing.MaximaDetection import findExtendedMaxima
dataMax = findExtendedMaxima(dataDoG, hMax = None, verbose = True, threshold = 10);
clrplt.plotOverlayLabel( dataDoG / dataDoG.max(), dataMax.astype('int'))

from ClearMap.ImageProcessing.MaximaDetection import findCenterOfMaxima
cells = findCenterOfMaxima(data, dataMax);
print cells.shape

clrplt.plotOverlayPoints(data, cells)

from ClearMap.ImageProcessing.CellSizeDetection import detectCellShape
dataShape = detectCellShape(dataDoG, cells, threshold = 15);
clrplt.plotOverlayLabel(dataDoG / dataDoG.max(), dataShape, z = (10,16))

from ClearMap.ImageProcessing.CellSizeDetection import findCellSize, findCellIntensity
cellSizes = findCellSize(dataShape, maxLabel = cells.shape[0]);
cellIntensities = findCellIntensity(dataBGR, dataShape, maxLabel = cells.shape[0]);

import matplotlib.pyplot as mpl
mpl.figure()
mpl.plot(cellSizes, cellIntensities, '.')
mpl.xlabel('cell size [voxel]')
mpl.ylabel('cell intensity [au]')

get_ipython().magic('matplotlib inline')

import math
import csv,gc
import matplotlib
import numpy as np
import cv2

#%matplotlib
BINS = 32

import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage import exposure
import cv2
from PIL import Image
from numpy import *

get_ipython().system('wget -O poorExample.png https://github.com/NeuroDataDesign/seelviz/blob/gh-pages/Tony/ipynb/poorExample.png?raw=true')
im = array(Image.open('poorExample.png'))
plt.imshow(im)

im = array(Image.open('poorExample.png').convert('L'))
plt.imshow(im, cmap='gray')

#Basic Thresholding

import Image
import numpy as np
import scipy.ndimage

image = np.asarray(im)
data = np.array(image)
threshold = 200
window = 10 # This is the "full" window...
new_value = 0

mask = data > threshold
mask = scipy.ndimage.uniform_filter(mask.astype(np.float), size=window)
mask = mask > 0
data[mask] = new_value

plt.imshow(np.asarray(data), cmap='gray')

#Global Histogram Equalization

equ = cv2.equalizeHist(im)
plt.imshow(equ, cmap='gray')

img = im

#Local Histogram Equalization

imgflat = img.reshape(-1)
print imgflat.sum()

print " "
fig = plt.hist(imgflat, bins=255)
plt.title('Histogram')
plt.show()

print " "

#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe = cv2.createCLAHE()

#img_grey = np.array(img * 255, dtype = np.uint8)
#threshed = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

cl1 = clahe.apply(img)
 
#cv2.imwrite('clahe_2.jpg',cl1)
#cv2.startWindowThread()
#cv2.namedWindow("adaptive")
#cv2.imshow("adaptive", cl1)
#cv2.imshow("adaptive", threshed)
#plt.imshow(threshed)

print " "

localimgflat = cl1.reshape(-1)
print localimgflat
print localimgflat.sum()

print " "
fig = plt.hist(localimgflat, bins=255)
plt.title('Locally Equalized Histogram')
plt.show()

plt.imshow(cl1)

plt.imshow(cl1, cmap = 'gray')

