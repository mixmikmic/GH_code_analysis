get_ipython().magic('matplotlib inline')
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#Open a video from our dataset
fileName = "../../data/UCF-101/Archery/v_Archery_g01_c01.avi"
cap = cv2.VideoCapture(fileName)

#Make sure the video opens
while not cap.isOpened():
    cap = cv2.VideoCapture(fileName)
    cv2.waitKey(1000)
    print "Wait for the header"
    
#Start reading frames
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

#Checking the number of frames in the video
length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

#We want 30 frames per video
tmp = length / 30

valid_frames = [tmp*i for i in xrange(0,length/tmp)]

print valid_frames

#Open a videowriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20, (x,y))

frame_num = 0

#Loop through all of the frames and calculate optical flow
while(1):
    frame_num += 1
    ret, frame2 = cap.read()
    
    #If no more frames can be read then break out of our loop
    if(not(ret)):
        break
        
    if(frame_num in valid_frames):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        out.write(rgb)

        prvs = next

#Close the file
cap.release()
out.release()

get_ipython().magic('matplotlib inline')
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#Open a video from our dataset
fileName = "../../data/UCF-101/Archery/v_Archery_g01_c01.avi"
cap = cv2.VideoCapture(fileName)

#Make sure the video opens
while not cap.isOpened():
    cap = cv2.VideoCapture(fileName)
    cv2.waitKey(1000)
    print "Wait for the header"
    
#Start reading frames
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

#Checking the number of frames in the video
length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

#We want 30 frames per video
tmp = length / 30

valid_frames = [tmp*i for i in xrange(0,length/tmp)]

print valid_frames

#Open a videowriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output_bw.avi',fourcc, 6, (x,y),0)

frame_num = 0

#Loop through all of the frames and calculate optical flow
while(1):
    frame_num += 1
    ret, frame2 = cap.read()
    
    #If no more frames can be read then break out of our loop
    if(not(ret)):
        break
        
    if(frame_num in valid_frames):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        bw = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        out.write(bw)

        prvs = next

#Close the file
cap.release()
out.release()



