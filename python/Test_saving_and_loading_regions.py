# setup
import numpy as np
import cv2
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from pprint import pprint

import salientregions as sr
import scipy.io as sio

get_ipython().magic('pylab inline')

# load the test image
testdata_images_path = '../tests/images/Binary/'
testdata_features_path = '../tests/features/Binary/'

image = cv2.imread(os.path.join(testdata_images_path, 'Binary_all_types_noise.png'), cv2.IMREAD_GRAYSCALE)
sr.show_image(image)

# load MATLAB SE and set up parameters and detector
SE = sio.loadmat(os.path.join(testdata_features_path,"SE_all.mat"))['SE_n']
lam = 50
area_factor = 0.05
connectivity = 4
binarydetector = sr.BinaryDetector(SE=SE, lam=lam, area_factor=area_factor, 
                                                connectivity=connectivity)

# run detector
regions = binarydetector.detect(image, find_holes=True, find_islands=True,
       find_indentations=True, find_protrusions=True, visualize=True)

# find the equivalent ellipses (both using standard and polynimial coefficients)
num_regions, features_standard, features_poly = sr.binary_mask2ellipse_features(regions, min_square=False)
print("Total number of regions detected: ", sum(num_regions.values()))
print("Number of features per saliency type: ", num_regions)
#sr.visualize_ellipses(regions["holes"], features_standard["holes"])
#sr.visualize_ellipses(regions["islands"], features_standard["islands"])
#sr.visualize_ellipses(regions["indentations"], features_standard["indentations"])
#sr.visualize_ellipses(regions["protrusions"], features_standard["protrusions"])
#sr.visualize_elements_ellipses(image, features_standard);

# print the feature representations
print("Elliptic standard features: \n")
pprint(features_standard) 
print("\n Elliptic polynomial features: \n")
pprint(features_poly)

# save the ellipsies in txt files
filename_standard = (os.path.join(testdata_features_path,'features_standard.txt'))
total_num_regions = sr.save_ellipse_features2file(num_regions, features_standard, filename_standard)
print("Total_num_regions (standard)", total_num_regions)
filename_poly = (os.path.join(testdata_features_path,'features_poly.txt'))
total_num_regions = sr.save_ellipse_features2file(num_regions, features_poly, filename_poly)
print("Total_num_regions (poly)", total_num_regions)

# loading the ellipses from the txt files
total_num_regions, num_regions, features_standard_loaded = sr.load_ellipse_features_from_file(filename_standard)
print("Features standard: ")
print("Total_num_regions: ", total_num_regions)
print("Number of features per saliency type: ", num_regions)

total_num_regions, num_regions, features_poly_loaded = sr.load_ellipse_features_from_file(filename_poly)
print("\n Features standard: ")
print("Total_num_regions: ", total_num_regions)
print("Number of features per saliency type: ", num_regions)

#comapre the loaded ellipses with the original ones
print("Comparing the original standard and the loaded features: \n")
if all(features_standard[k]==features_standard_loaded[k] for k in features_standard):
    print("The same! \n")
else:
    print("Different! \n")

print("Comparing the original polynomial and the loaded features: \n")
if all(features_poly[k]==features_poly_loaded[k] for k in features_poly):
    print("The same! \n")
else:
    print("Different! \n")

# visualise the original and the  loaded (only standard is supported!)
sr.visualize_elements_ellipses(image, features_standard);
sr.visualize_elements_ellipses(image, features_standard_loaded);
#sr.visualize_elements_ellipses(image, features_poly);



