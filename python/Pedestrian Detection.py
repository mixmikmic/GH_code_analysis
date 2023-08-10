from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import os
import imutils
import cv2

dirs = os.listdir('Images')
for i in range(0,len(dirs)):
    dirs[i] = os.path.join('Images', dirs[i])
paths = []
for dir in dirs:
    files = os.listdir(dir)
    for file in files:
        paths.append(os.path.join(dir,file))


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
for imagePath in paths:
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    print(imagePath)
    image = cv2.imread(imagePath)
    print(image)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
 
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)
 
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
    # show some information on the number of bounding boxes
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))
 
    # show the output images
    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    # Press any key to go on to the next image
    cv2.waitKey(0)

