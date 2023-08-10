import os
import glob

import pandas as pd
import numpy as np
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from PIL import Image

images_folder_path = './data/fisheye_camera_calibration/'
img_list = glob.glob(os.path.join(images_folder_path,'*.jpg'))
img = Image.open(os.path.join(img_list[0]))
img

cb_w, cb_h =  CHECKERBOARD_SIZE = (7,9)

h, w = img_size = np.array(img).shape[:2]
img_size

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cb_w*cb_h, 3), np.float32)
objp[:,:2] = np.mgrid[0:cb_w,0:cb_h].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in img_list[0:50]:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners2,ret)
        
plt.imshow(img)

#Get the camera properties.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                   gray.shape[::-1],None,None)

#ret
ret

#camera matrix
np.round(mtx)

#distance coefficients
np.round(dist)

#Scale coeficients to the image and unit vector 1.
img = cv2.imread(img_list[2])
h,  w = img.shape[:2]
scaled_mtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, scaled_mtx)
plt.imshow(dst)

#not sure what this does
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

"""
TODO: these next two stips could be the same as above 
method but the objpoints is a different shape.
"""
#create a new object point array
objp = np.zeros((1, cb_w*cb_h, 3), np.float32)
objp[0,:,:2] = np.mgrid[0:cb_w, 0:cb_h].T.reshape(-1, 2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#generate img and object point points
for fname in img_list:
    
    # Find the chess board corners
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
        
img_count = len(imgpoints)
print("Found {} valid images for calibration".format(img_count))



#Calculate the camera properties.
mtx= np.zeros((3, 3))
dist = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(img_count)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(img_count)]

rms, _, _, _, _ =     cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        img_size,
        mtx,
        dist,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

mtx

#camera matrix
np.round(mtx)

#distribution coefficients
dist

def undistort(img_arr):
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), 
                                                     mtx, (w, h), cv2.CV_16SC2)

    undistorted_img = cv2.remap(img_arr, map1, map2, interpolation=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img


#undistort image
img = Image.open(img_list[60])
undistorted = undistort(np.array(img))

#see difference between the two
fig = plt.figure()
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(img)
a.set_title('Original')
a=fig.add_subplot(1,2,2)
a.set_title('Undistorted')
imgplot = plt.imshow(undistorted)

#scale to image_size = 120x160 (4x smaller)
scaled_mtx = mtx/4
scaled_mtx


focal_length = fx, fy = scaled_mtx[0,0], scaled_mtx[1,1]
focal_length

optical_axis_center = cx, cy = scaled_mtx[0,2], scaled_mtx[1,2]
optical_axis_center

#distortion coefficients
k1, k2, k3, k4 = dist #
dist







