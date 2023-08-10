import os
import math
#import random

import numpy as np
import tensorflow as tf
import cv2
import os

from imutils.video import WebcamVideoStream

slim = tf.contrib.slim

get_ipython().magic('pylab inline')
from IPython.display import clear_output

#%matplotlib inline
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#from skimage import io
import time
import subprocess

precision = 10
from datetime import datetime

def getCurrentClock():
    #return time.clock()
    return datetime.now()

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NCHW'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

import time
# Test on some demo image and visualize output.
#path = '../demo/'
#image_names = sorted(os.listdir(path))

#img = io.imread("http://www.searchamateur.com/pictures/street-cars-second-life.jpg") #not all detected

start_time = time.time()
#rclasses, rscores, rbboxes =  process_image(img)
# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
#visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
print("--- %s seconds ---" % (time.time() - start_time))

#sys.path.append('c:/windows/system32')
#A smooth drive in The Crew on PS4 - OSSDC Simulator ACC Train 30fps

#videoUrl = subprocess.Popen("c:/windows/system32/youtube-dl.exe -f22 -g https://www.youtube.com/watch?v=uuQlMCMT71I", shell=True, stdout=subprocess.PIPE).stdout.read()

'''
p = subprocess.Popen(['c:/windows/system32/youtube-dl.exe', '-f22', '-g' ,'https://www.youtube.com/watch?v=uuQlMCMT71I'], stdout=subprocess.PIPE, bufsize=0)

for line in iter(p.stdout.readline, b''):
    print ('>>> {}'.format(line.rstrip()))
    
line=''
while True:
    line = p.stdout.readline()
    if not line: 
        break
    print (line)
    
videoUrl = line
'''

videoUrl = "https://r3---sn-ux3n588t-mjh6.googlevideo.com/videoplayback?pl=17&ratebypass=yes&ei=Lol5WfanNMeuugLswZvACw&ipbits=0&dur=182.950&sparams=dur%2Cei%2Cid%2Cinitcwndbps%2Cip%2Cipbits%2Citag%2Clmt%2Cmime%2Cmm%2Cmn%2Cms%2Cmv%2Cpl%2Cratebypass%2Crequiressl%2Csource%2Cexpire&itag=22&requiressl=yes&expire=1501158799&mime=video%2Fmp4&id=o-AEbewc67rnUgvDdK1tuFOt4j3vzh28EinR-vA4A6_MO4&mn=sn-ux3n588t-mjh6&mm=31&signature=3A68C8CE4277F7DEA920F8B1093CE5C2755DB4A4.53E1CCF9FE284E2CBA14B7A7A9B0B1CCF0F19FBB&initcwndbps=1658750&lmt=1471096522525866&key=yt6&ip=24.212.175.55&ms=au&mt=1501137045&mv=m&source=youtube"
#youtube-dl.exe -f22 -g https://www.youtube.com/watch?v=txg6RMEYzE4
#videoUrl = videoUrl.decode("utf-8").rstrip()
print("videoUrl =",videoUrl)

webcam=False
#webcam=True

if webcam:
    cap = WebcamVideoStream(videoUrl).start()
else:
    cap = cv2.VideoCapture(videoUrl)

count=50
skip=2000
SKIP_EVERY=150 #pick a frame every 5 seconds

count=1000
skip=1000 #int(7622-5)
SKIP_EVERY=0

every=SKIP_EVERY
initial_time = getCurrentClock()
flag=True

frameCnt=0
prevFrameCnt=0
prevTime=getCurrentClock()

showImage=False
showImage=True
processImage=False
processImage=True
zoomImage=0
rclasses = []
rscores = []
rbboxes = []

record = False
#record = True

procWidth = 1280 #640   # processing width (x resolution) of frame
procHeight = 720   # processing width (x resolution) of frame

out = None
if record:
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    out = cv2.VideoWriter('output-'+timestr+'.mp4',fourcc, 30.0, (int(procWidth),int(procHeight)))
    
try:
    while True:
        #frame = cap.read()
        #if True:
        if webcam or cap.grab():
            if webcam:
                frame = cap.read()
            else:
                flag, frame = cap.retrieve()    
            if not flag:
                continue
            else:
                frameCnt=frameCnt+1
                nowMicro = getCurrentClock()
                delta = (nowMicro-prevTime).total_seconds()
                #print("%f " % (delta))
                if delta>=1.0:
                    #print("FPS = %0.4f" % ((frameCnt-prevFrameCnt)/delta))
                    prevTime = nowMicro
                    prevFrameCnt=frameCnt

                if skip>0:
                    skip=skip-1
                    continue

                if every>0:
                    every=every-1
                    continue
                every=SKIP_EVERY

                count=count-1
                if count==0:
                    break

                img = frame
                if processImage:    
                    if zoomImage>0:
                        #crop center of image, crop width is output_side_length
                        output_side_length = int(1920/zoomImage)
                        height, width, depth = frame.shape
                        #print (height, width, depth)
                        height_offset = int((height - output_side_length) / 2)
                        width_offset = int((width - output_side_length) / 2)
                        #print (height, width, depth, height_offset,width_offset,output_side_length)
                        img = frame[height_offset:height_offset + output_side_length,width_offset:width_offset + output_side_length]

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    start_time = getCurrentClock()
                    rclasses, rscores, rbboxes =  process_image(img)
                    if len(rclasses)>0:
                        nowMicro = getCurrentClock()
                        print("# %s - %s - %0.4f seconds ---" % (frameCnt,rclasses.astype('|S3'), (nowMicro - start_time).total_seconds()))
                        start_time = nowMicro
                    if showImage:
                        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
                if showImage:
                    #visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
                    #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
                    #if processImage:
                        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    #cv2.imshow("ssd",img)
                    imshow(img)
                    show()
                    # Display the frame until new frame is available
                    clear_output(wait=True)
                if record:
                    #if processImage:
                        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    newimage = cv2.resize(img,(procWidth,procHeight))
                    out.write(newimage)
        key = cv2.waitKey(1)
        if  key == 27:
            break
        elif key == ord('u'):
            showImage= not(showImage)
        elif key == ord('p'):
            processImage= not(processImage)
        elif key == ord('z'):
            zoomImage=zoomImage+1
            if zoomImage==10:
                zoomImage=0
        elif key == ord('x'):
            zoomImage=zoomImage-1
            if zoomImage<0:
                zoomImage=0
except KeyboardInterrupt:
    # Release the Video Device
    vid.release()
    # Message to be displayed after releasing the device
    print ("Released Video Resource")
    
nowMicro = getCurrentClock()
print("# %s -- %0.4f seconds - FPS: %0.4f ---" % (frameCnt, (nowMicro - initial_time).total_seconds(), frameCnt/(nowMicro - initial_time).total_seconds()))

import subprocess
def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
# Example
for line in execute(["youtube-dl.exe","-f22 -g https://www.youtube.com/watch?v=uuQlMCMT71I"]):
    print(line, end="")







