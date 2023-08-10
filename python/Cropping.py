get_ipython().magic('matplotlib inline')
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

fileName = "../../data/UCF-101/Archery/v_Archery_g01_c01.avi"

print "Now processing: "+fileName.split("/")[5]
cap = cv2.VideoCapture(fileName)

#Checking the number of frames in the video
length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
x = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
y = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

#Open a videowriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
newFileName = fileName.split('/')[5].split('.')[0]+"_cropped.avi"
out = cv2.VideoWriter(newFileName ,fourcc, 20, (x/2,y/2))

while(1):
    ret, frame = cap.read()

    #If no more frames can be read then break out of our loop
    if(not(ret)):
        break

    frame_height, frame_width, RGB = frame.shape

    height_offset = frame_height/4
    width_offset = frame_width/4

    #Take the center part of the frame
    new_frame = frame[height_offset:(y/2)+height_offset, width_offset:(x/2)+width_offset, :]

    #Output to new avi file
    out.write(new_frame)

# Release everything if job is finished
cap.release()
out.release()



