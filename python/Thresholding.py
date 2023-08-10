import cv2
import numpy as np

#only for TrackBar process
def nothing(x):
    pass

image_org = cv2.imread('data/logo.jpg')
image = cv2.cvtColor(image_org,cv2.COLOR_RGB2GRAY)

maxval = 255
thresh=0
type_thresh = 2
cv2.namedWindow("Adjust",cv2.WINDOW_AUTOSIZE); #Threshold settings window
cv2.createTrackbar("Thresh", "Adjust", thresh, 200, nothing);
cv2.createTrackbar("Max", "Adjust", maxval, 255, nothing);

#Threshold methods correspond integer numbers in OpenCV Library,(binary threshold,otsu threshold etc)
#And threshold methods summable with each other like; cv2.BINARY_THRESH + cv2.OTSU_THRESH or 1 + 4
cv2.createTrackbar("Type", "Adjust", type_thresh, 4, nothing); 

Threshold = np.zeros(image.shape, np.uint8)

# Infinite loop until we hit the escape key on keyboard
while 1:
    thresh = cv2.getTrackbarPos('Thresh', 'Adjust')
    maxval = cv2.getTrackbarPos('Max', 'Adjust')
    type_thresh = cv2.getTrackbarPos('Type', 'Adjust')
    retval,Threshold = cv2.threshold(image,thresh,maxval,type_thresh)
    # display images
    cv2.imshow('Adjust', Threshold)
    cv2.imshow('Original', image_org)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:   # hit escape to quit
        break
        
cv2.destroyAllWindows()        



