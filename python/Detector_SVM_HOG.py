# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
from skvideo.io import FFmpegWriter 

import time

def pedestrian_detection(input_video, outfile, fps=30):
    vidcap = cv2.VideoCapture(input_video)
   
    writer = FFmpegWriter(outfile, outputdict={'-r': fps})
    writer = FFmpegWriter(outfile)
    
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    start_time = time.time()
    i = 0
    while True:
        ret, image = vidcap.read()
        if not ret: break
        i += 1
        image = imutils.resize(image, width=(image.shape[1]))
        orig = image.copy()

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        
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
            print ("Found human in frame", i)
        writer.writeFrame(image)
    print("--- %s seconds ---" % (time.time() - start_time))
    writer.close()

pedestrian_detection('../../videos/uber_trimmed.mp4', './opencv-ped-detect.mp4')

#src = '../PedestrianPhotos-Orig/'
#dest = '../PedestrainPhotos-Boxes/'

src='../Frames/'
dest='../FramesPed-Boxes/'
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def run_hog_detector(src, dest):
    # loop over the image paths
    for imagePath in src:
        print (imagePath)
        # load the image and resize it to (1) reduce detection time
        # and (2) improve detection accuracy
        image = cv2.imread(imagePath)
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
        filename = "hog_" + imagePath
        print("[INFO] {}: {} original boxes, {} after suppression".format(
            filename, len(rects), len(pick)))

        # show the output images
        #cv2.imwrite('%s.jpg' %filename , image)
        cv2.imwrite(('{0}{1}').format(dest, filename), image)

run_hog_detector(["night_ped_5.jpg"], "./")

def images_seq_to_video(input_folder, outfile, count, fps=30):
    writer = FFmpegWriter(outfile, outputdict={'-r': fps})
    writer = FFmpegWriter(outfile)
    
    for i in range(count):
        image = input_folder + 'frame'+ str(i)+'.jpg'
        f = cv2.imread(image)
        plt.imshow(f)
        writer.writeFrame(f)
    writer.close()

input_folder = '../Frames/'
outfile = 'uber_opencv_pedestrian_detection.mp4'
images_seq_to_video(input_folder, outfile, 166)



