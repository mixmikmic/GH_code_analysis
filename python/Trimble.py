## Imports

import cv2
import json
import matplotlib.pyplot as pyplot
import numpy
import matplotlib.patches  as patches
import pynq.overlays.base
import pynq.lib.arduino as arduino
import pynq.lib.video as video
import requests
import scipy
import sys
import PIL

## Config
gAWS_TENSORFLOW_INSTANCE = 'http://34.202.159.80'
gCAMERA0_IMAGE = "/home/xilinx/jupyter_notebooks/trimble-mp/CAM2_image_0032.jpg"
gCAMERA1_IMAGE = "/home/xilinx/jupyter_notebooks/trimble-mp/CAM3_image_0032.jpg"

base = pynq.overlays.base.BaseOverlay("base.bit")
hdmi_in = base.video.hdmi_in
hdmi_out = base.video.hdmi_out
v = video.VideoMode(1920,1080,24)
hdmi_out.configure(v, video.PIXEL_BGR)
hdmi_out.start()
outframe = hdmi_out.newframe()

# Read images
image0BGR = cv2.imread(gCAMERA0_IMAGE)
image1BGR = cv2.imread(gCAMERA1_IMAGE)

image0 = image0BGR[...,::-1]
image1 = image1BGR[...,::-1]

# Show image 0 on HDMI

# Need to resize it first
outframe[:] = cv2.resize(image0BGR, (1920, 1080));
hdmi_out.writeframe(outframe)

## Show image on LCD

# Open LCD object and clear
lcd = arduino.Arduino_LCD18(base.ARDUINO)
lcd.clear()

# Write image to disk
nw = 160
nl = 128
cv2.imwrite("/home/xilinx/small.jpg", cv2.resize(image0BGR, (nw,nl)))

# Display!
lcd.display("/home/xilinx/small.jpg",x_pos=0,y_pos=127,orientation=3,background=[255,255,255])

def RemoteTensorFlowClassify(image_name_string):
    f = open(image_name_string,'rb')
    r = requests.put(gAWS_TENSORFLOW_INSTANCE, data=f)
    return json.loads(r.content.decode())

#Return the object that camera zero sees with the maximum score
cam0_json_return = RemoteTensorFlowClassify(gCAMERA0_IMAGE)
json0 = cam0_json_return["image_detection"]
max = 0.0
out = []
for var in json0['object']:
    if (var['score'] > max):
        out = var
json0 = out
json0

#Return the object that camera one sees with the maximum score
cam1_json_return = RemoteTensorFlowClassify(gCAMERA1_IMAGE)
json1 = cam1_json_return["image_detection"]
max = 0.0
out = []
for var in json1['object']:
    if (var['score'] > max):
        out = var
json1 = out
json1

def DrawRect(the_json,the_image, x1, x2, y1, y2 ): 
    # Currently offline until the TesnorFlow net is fixed
    #x1 = int(the_json["xmin"]) 
    #y1 = int(the_json["ymin"]) 
    #x2 = int(the_json["xmax"]) 
    #y2 = int(the_json["ymax"])
 
    
    fig, ax = pyplot.subplots(1)
    ax.imshow(the_image)
    rect = patches.Rectangle((x1,y1), (x2-x1), (y2-y1), linewidth = 1 , edgecolor = 'r', facecolor='none') 
    ax.add_patch(rect)
    pyplot.show()

## Convert to grayscale
grayImage0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
grayImage1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

def IsInsideROI(pt, the_json, x1, x2, y1, y2):
#    x_min = int(the_json["object"]["xmin"])
#    y_min = int(the_json["object"]["ymin"])
#    x_max = int(the_json["object"]["xmax"])
#    y_max = int(the_json["object"]["ymax"])
 
    x_min = x1
    y_min = y1
    x_max = x2
    y_max = y2
    if(pt[0]>=x_min and pt[0] <=x_max and pt[1]>=y_min and pt[1]<=y_max):
        return True
    else: 
        return False

## Detect keypoints
Brisk = cv2.BRISK_create()

keyPoints0 = Brisk.detect(grayImage0)
keyPoints1 = Brisk.detect(grayImage1)

## Find keypoints inside ROI
roiKeyPoints0 = numpy.asarray([k for k in keyPoints0 if IsInsideROI(k.pt,json0, 955, 1045, 740, 1275 )])
roiKeyPoints1 = numpy.asarray([k for k in keyPoints1 if IsInsideROI(k.pt,json1, 1335, 1465, 910, 1455 )])

## Compute descriptors for keypoitns inside ROI
[keyPoints0, desc0] = Brisk.compute(grayImage0, roiKeyPoints0);
[keyPoints1, desc1] = Brisk.compute(grayImage1, roiKeyPoints1);

## Find matches of ROI keypoints
BF = cv2.BFMatcher()
matches = BF.match(desc0, desc1)

## Extract pixel coordinates from matched keypoints

x_C0 = numpy.asarray([keyPoints0[match.queryIdx].pt for match in matches])
x_C1 = numpy.asarray([keyPoints1[match.trainIdx].pt for match in matches])

# Triangulate points

# We need projection matrices for camera 0 and camera 1
f = 8.350589e+000 / 3.45E-3
cx = -3.922872e-002 / 3.45E-3
cy = -1.396717e-004 / 3.45E-3
K_C0 = numpy.transpose(numpy.asarray([[f, 0, 0], [0, f, 0], [cx, cy, 1]]))
k_C0 = numpy.asarray([1.761471e-003, -2.920431e-005, -8.341438e-005, -9.470247e-006, -1.140118e-007])

[R_C0, J] = cv2.Rodrigues(numpy.asarray([1.5315866633, 2.6655790203, -0.0270418317]))
T_C0 = numpy.transpose(numpy.asarray([[152.9307390952, 260.3066944976, 351.7405264829]])) * 1000

f = 8.259861e+000 / 3.45E-3
cx = 8.397453e-002 / 3.45E-3
cy = -2.382030e-002 / 3.45E-3
K_C0 = numpy.transpose(numpy.asarray([[f, 0, 0], [0, f, 0], [cx, cy, 1]]))
K_C1 = numpy.asarray([1.660053e-003, -2.986269e-005, -7.461966e-008, -2.247960e-004, -2.290483e-006])

[R_C1, J] = cv2.Rodrigues(numpy.asarray([1.4200199799, -2.6113619450, -0.1371719827]))
T_C1 = numpy.transpose(numpy.asarray([[146.8718203137, 259.9661037150, 351.5832136366]])) * 1000

P_C0 = numpy.dot(K_C0,numpy.concatenate((R_C0, T_C0), 1))
P_C1 = numpy.dot(K_C1,numpy.concatenate((R_C1, T_C1), 1))

# Compute 3D coordinates of detected points
X_C0 = cv2.convertPointsFromHomogeneous(numpy.transpose(cv2.triangulatePoints(P_C0, P_C1, numpy.transpose(x_C0), numpy.transpose(x_C1))))



