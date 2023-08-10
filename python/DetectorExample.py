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

det = sr.SalientDetector(SE_size_factor=0.20,
                                lam_factor=4)

regions = det.detect(img,
                    find_holes=True,
                    find_islands=True,
                    find_indentations=True,
                    find_protrusions=True, 
                    visualize=True)
print(regions.keys())

num_regions, features_standard, features_poly = sr.binary_mask2ellipse_features(regions, min_square=False)
print("number of features per saliency type: ", num_regions)
sr.visualize_ellipses(regions["holes"], features_standard["holes"])
sr.visualize_ellipses(regions["islands"], features_standard["islands"])
sr.visualize_ellipses(regions["indentations"], features_standard["indentations"])
sr.visualize_ellipses(regions["protrusions"], features_standard["protrusions"])
#print "Elliptic polynomial features:",  features_poly
sr.visualize_elements_ellipses(img, features_standard);

total_num_regions = sr.save_ellipse_features2file(num_regions, features_poly, 'poly_features.txt')
print("total_num_regions", total_num_regions)

import sys, os
sys.path.insert(0, os.path.abspath('..'))
import salientregions as sr
total_num_regions, num_regions, features = sr.load_ellipse_features_from_file('poly_features.txt')
print("total_num_regions: ", total_num_regions)
print("number of features per saliency type: ", num_regions)
#print "features: ", features

