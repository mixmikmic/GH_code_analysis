# In this cell, some libraries were imported
import cv2
import sys
import os
from PIL import Image, ImageDraw
import pylab
import time

# Face Detection Function
def detectFaces(image_name):
    print ("Face Detection Start.")
    # Read the image and convert to gray to reduce the data
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Color => Gray
    
    # The haarcascades classifier is used to train data
    #face_cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)#1.3 and 5are the min and max windows of the treatures
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    print ("Face Detection Complete.")
    return result

#Crop faces and save them in the same directory
filepath ="/home/xilinx/jupyter_notebooks/OpenCV/Face_Detection/images/"
dir_path ="/home/xilinx/jupyter_notebooks/OpenCV/Face_Detection/"
filecount = len(os.listdir(filepath))-1
image_count = 1#count is the number of images
face_cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")
for fn in os.listdir(filepath): #fn 表示的是文件名
    start = time.time()
    if image_count <= filecount:
        image_name = str(image_count) + '.JPG'
        image_path = filepath + image_name
        image_new = dir_path + image_name
        #print (image_path)
        #print (image_new)
        os.system('cp '+(image_path)+ (' /home/xilinx/jupyter_notebooks/OpenCV/Face_Detection/'))
        faces = detectFaces(image_name)
        if not faces:
            print ("Error to detect face")
        if faces:
            #All croped face images will be saved in a subdirectory
            face_name = image_name.split('.')[0]
            #os.mkdir(save_dir)
            count = 0
            for (x1,y1,x2,y2) in faces:
                file_name = os.path.join(dir_path,face_name+str(count)+".jpg")
                Image.open(image_name).crop((x1,y1,x2,y2)).save(file_name)
                #os.system('rm -rf '+(image_path)+' /home/xilinx/jupyter_notebooks/OpenCV/Face_Detection/')
                count+=1    
            os.system('rm -rf '+(image_new))
            print("The " + str(image_count) +" image were done.")
            print("Congratulation! The total of the " + str(count) + " faces in the " +str(image_count) + " image.")
    end = time.time()
    TimeSpan = end - start
    if image_count <= filecount:
        print ("The time of " + str(image_count) + " image is " +str(TimeSpan) + " s.")
    image_count = image_count + 1   

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pylab

# Initiate ORB detector
orb = cv2.ORB_create()

img1 = cv2.imread('/home/xilinx/jupyter_notebooks/OpenCV/Face_Detection/20.jpg',cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('/home/xilinx/jupyter_notebooks/OpenCV/Face_Detection/30.jpg',cv2.COLOR_BGR2GRAY)

#plt.imshow(img1),plt.show()
#plt.imshow(img2),plt.show()

brisk = cv2.BRISK_create()
(kpt1, desc1) = brisk.detectAndCompute(img1, None)  
bk_img1 = img1.copy()  
out_img1 = img1.copy()  
out_img1 = cv2.drawKeypoints(bk_img1, kpt1, out_img1)
plt.imshow(out_img1),plt.show()

(kpt2, desc2) = brisk.detectAndCompute(img1, None)  
bk_img2 = img2.copy()  
out_img2 = img2.copy()  
out_img2 = cv2.drawKeypoints(bk_img2, kpt2, out_img2)  
plt.imshow(out_img2),plt.show()

# 特征点匹配  
matcher = cv2.BFMatcher()  
matches = matcher.match(desc1, desc2)  
print(matches) 

matches = sorted(matches, key = lambda x:x.distance)

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out

# Draw first 10 matches.
print (len(matches))
img3 = drawMatches(img1,kpt1,img2,kpt2,matches[:2])

plt.imshow(img3),plt.show()



