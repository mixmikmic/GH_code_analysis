import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from collections import deque
from featureExtraction import *

# Camera Calibration
import os
import cameraCalibration
(ret, cameraMat, distCoeffs, rvecs, tvecs), fig = cameraCalibration.get_calibration_matrix(os.path.join('camera_cal','*.jpg'))
plt.close()

# Perspective Transform Params
img_size = (1280, 720)
width, height = img_size
offset = 200
src = np.float32([
    [  588,   446 ],
    [  691,   446 ],
    [ 1126,   673 ],
    [  153 ,   673 ]])
dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# Lane Detection Pipeline
from lane_pipeline import LanePipeline
from line import Line
line=Line()
LanePipeline.set_values(line, M, Minv, cameraMat, distCoeffs)



def process_image(img):
    lane_detected = LanePipeline.pipeline(img)

from moviepy.editor import VideoFileClip
from IPython.display import HTML

white_output = 'project_lane_video_output.mp4'
clip = VideoFileClip("project_video.mp4")#.subclip(t_start=30,t_end=35)
white_clip = clip.fl_image(process_image)
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

