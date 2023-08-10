import os
import math
import random
import time 

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
data_format = 'NHWC'
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

pascal_classes = {0:"background",1:"aeroplane",2:"bicycle",3:"bird",4:"boat",5:"bottle",6:"bus",7:"car",8:"cat",9:"chair",10:"cow",11:"diningtable",12:"dog",13:"horse",14:"motorbike",15:"person",16:"pottedplant",17:"sheep",18:"sofa",19:"train",20:"tvmonitor"}

colors = dict()
def plt_bboxes(img, classes, scores, bboxes, figsize=(17.78,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    print ("original height width", height, width)
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = pascal_classes[cls_id]
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
            print ("Frame has class", class_name)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    print("Processed data")
    return data

def run_ssd_on_video(video_path, video_name):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = "out_" + video_name + "_SSD" + ".avi"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (1280,720)) #width, height format
    i = 0
    people_frames = []
    while(cap.isOpened()):
        has_frame, frame = cap.read()
        if not has_frame: 
            break
        # Custom code here 
        i += 1
        print ("Processing frame #", i)
        rclasses, rscores, rbboxes =  process_image(frame)
        print (rclasses)
        if (15 in rclasses):
            people_frames.append(i)
        out.write(plt_bboxes(frame, rclasses, rscores, rbboxes))  
    print("Finished reading video")
    print ("Wrote video to ", out_path)
    print ("Frames with person", people_frames)
    cap.release()
    out.release()
    time_taken = time.time() - start_time
    print("Time taken = ", time_taken)
    print("FPS = ", str(float(i/time_taken)))

# Test on some demo image and visualize output.
path = '../../Frames/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[-1])
rclasses, rscores, rbboxes =  process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
x = plt_bboxes(img, rclasses, rscores, rbboxes)

run_ssd_on_video("../../uber_trimmed.mp4", "uber_trimmed_new_test")

img = cv2.imread("../../Frames/frame89.jpg")
rclasses, rscores, rbboxes =  process_image(img)
x = plt_bboxes(img, rclasses, rscores, rbboxes, (10,10), 1.5)
# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)

run_ssd_on_video("Crashes/crash1.mp4", "crash1")
run_ssd_on_video("Crashes/crash2.mp4", "crash2")
run_ssd_on_video("Crashes/crash3.mp4", "crash3")
run_ssd_on_video("Crashes/crash4.mp4", "crash4")
run_ssd_on_video("Crashes/crash5.mp4", "crash5")
run_ssd_on_video("Crashes/crash6.mp4", "crash6")
run_ssd_on_video("Crashes/crash7.mp4", "crash7")
run_ssd_on_video("Crashes/crash8.mp4", "crash8")
run_ssd_on_video("Crashes/crash9.mp4", "crash9")
run_ssd_on_video("Crashes/crash10.mp4", "crash10")
run_ssd_on_video("Crashes/crash11.mp4", "crash11")

