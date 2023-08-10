import cv2
import numpy as np

# Checking the OpenCV version
print("OpenCV version", cv2.__version__)

# Checking the Numpy version
print("Numpy version", np.__version__)

import numpy as np
import cv2
import os
import glob
from matplotlib import pyplot as plt

get_ipython().magic('matplotlib inline')

# Using the sample images provided with OpenCV
PATTERN_PATH = os.path.join("/","opencv","samples","data")

# we will be using this as a corner refinement criteria, while detecting chessboard corners
criteria = (cv2.TERM_CRITERIA_EPS  + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# preparing object points with Z = 0
objp = np.zeros((6*7,3), np.float32)

objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

obj_points = []
image_points = []

for image in glob.glob(os.path.join(PATTERN_PATH,"left*.jpg")):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

    if ret == True:
        obj_points.append(objp)
        
        # Further refines the corners detected in the images, by setting up a custom refinement criteria
        # as we have passed
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        image_points.append(corners2)

# At this stage we have both the object points and the image points
# next, we will need to calibrate the camera using the image and the object points , to get the camera matrix

# Using some random image from 1-12
img = cv2.imread(os.path.join(PATTERN_PATH,"left12.jpg"))
h, w = img.shape[:2]

# Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, image_points, (h,w), None, None)

# Rendering the Intrinsic Camera matrix
print("Intrinsic Camera Matrix")
print("=======================")
print(mtx)

print("Camera Centers")
print("=======================")
print("Cx: {0} Cy: {1}".format(mtx[0][2],mtx[1][2]))

print("Focal Lengths")
print("=======================")
print("Fx: {0} Fy: {1}".format(mtx[0][0],mtx[1][1]))

# Rendering the Extrinsic Camera matrix
print("Extrinsic Camera Parameters")
print("")

print("Translation Matrix")
print("=======================")
print(tvecs)

print("Rotational Matrix")
print("=======================")
print(rvecs)



