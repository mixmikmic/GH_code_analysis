import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
#import pickle
from find_cars import *
from hog_util_functions import draw_boxes

import imageio
#imageio.plugins.ffmpeg.download()
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

frame_ind = 0;
pickle_file = "HOGClassifier.p"
load_classifier(pickle_file)

# Tuning & saving heatmaps
avgBoxes = BoundingBoxes(10)

def process_image(image):
    global frame_ind    
    frame_ind += 1
    result = process_image_hog_pipeline(image, frame_ind, useHeatmap=True, thresh=1,
                                        avgBoxes=avgBoxes, verbose=True, verboseSaveHeatmaps=True)      
    return result

# Generate final version
avgBoxes = BoundingBoxes(10)
verbose = True

def process_image(image):
    global frame_ind
    global avgBoxes
    global verbose
    
    frame_ind += 1
    
    result = process_image_hog_pipeline(image, frame_ind, useHeatmap=True, thresh=29,
                                        avgBoxes=avgBoxes, verbose=verbose, verboseSaveHeatmaps=False)      
    return result

frame_ind = 0
out_dir='./output_images/'
output = out_dir + 'processed_project_video.mp4'
clip = VideoFileClip("project_video.mp4")
#output = out_dir + 'processed_test_video.mp4'
#clip = VideoFileClip("test_video.mp4")
out_clip = clip.fl_image(process_image) 
get_ipython().magic('time out_clip.write_videofile(output, audio=False)')

frame_ind = 0;

def process_image_grid(img):
    global frame_ind
    frame_ind += 1
    
    ystart = 400
    ystop = 656
    scale = 1
    
    #bboxes = find_cars_grid_160(img)
    #img = draw_boxes(img, bboxes, color=(0, 255, 255), thick=10)
    
    #bboxes = find_cars_grid_128(img)
    #img = draw_boxes(img, bboxes, color=(0, 0, 255), thick=8)

    #bboxes = find_cars_grid_96(img)
    #img = draw_boxes(img, bboxes, color=(0, 255, 0), thick=4)

    # Scale 1
#    bboxes = find_cars_grid_1(img)
#    result = draw_boxes(img, bboxes, color=(0, 0, 255), thick=4)
#    bboxes = find_cars_grid_1(img, cells_per_step=2)
#    result = draw_boxes(result, bboxes, color=(255, 0, 0), thick=1)

    # Scale 2
#    bboxes = find_cars_grid_2(img)
#    result = draw_boxes(img, bboxes, color=(0, 0, 255), thick=4)
#    bboxes = find_cars_grid_2(img, cells_per_step=2)
#    result = draw_boxes(result, bboxes, color=(255, 0, 0), thick=1)

    # Scale 3
    bboxes = find_cars_grid_3(img)
    result = draw_boxes(img, bboxes, color=(0, 0, 255), thick=4)
    bboxes = find_cars_grid_3(img, cells_per_step=2)
    result = draw_boxes(result, bboxes, color=(255, 0, 0), thick=1)
    
    
    # add frame_index text at the bottom of board
    xmax = 800
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'frame {:d}'.format(frame_ind), (xmax + 20, 60), font, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    
    return result

out_dir='./'
output = out_dir + 'grid_project_video.mp4'
clip = VideoFileClip("project_video.mp4")
#output = out_dir + 'grid_challenge_video.mp4'
#clip = VideoFileClip("challenge_video.mp4")
out_clip = clip.fl_image(process_image_grid) 
get_ipython().magic('time out_clip.write_videofile(output, audio=False)')

import matplotlib.pyplot as plt

out_dir='./temp_data/frames/project_video/'
frame_n = 0

def extract_all_frames(img):
    global out_dir
    global frame_n    
    frame_n += 1
    file_path = out_dir + str(frame_n) + '.jpg'
    #print(file_path)
    mpimg.imsave(file_path, img)
    return img

output = './temp_data/' + 'tmp_video.mp4'
clip = VideoFileClip("project_video.mp4")
out_clip = clip.fl_image(extract_all_frames) 
get_ipython().magic('time out_clip.write_videofile(output, audio=False)')



