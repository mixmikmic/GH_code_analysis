import numpy as np
import cv2
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import salientregions as sr

get_ipython().magic('pylab inline')

#Load the image
path_to_image = 'images/graffiti.jpg'
img = cv2.imread(path_to_image)
sr.show_image(img)

#Convert to grey scale
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sr.show_image(grayscale)

#Binarize with OTSU
binarized = sr.OtsuBinarizer().binarize(grayscale, visualize=True)

#Binarize according to threshold
thres = 128
binarizer = sr.ThresholdBinarizer(thres)
binarized = binarizer.binarize(grayscale, visualize=True)

img = grayscale
lam_factor = 3
area_factor_large = 0.001
area_factor_verylarge = 0.1
lam = 50
connectivity = 4
weights=(0.33,0.33,0.33)

binarizer = sr.DatadrivenBinarizer(area_factor_large=area_factor_large, area_factor_verylarge=area_factor_verylarge, 
                                           lam=lam, weights=weights, connectivity=connectivity)
binarized = binarizer.binarize(grayscale)



