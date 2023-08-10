import cv2
import dlib
import numpy as np

print("Checking the opencv version")
print(cv2.__version__)

print("Checking the dlib version")
print(dlib.__version__)

print("Checking the numpy version")
print(np.__version__)

import matplotlib.pyplot as plt
import matplotlib.image  as mpimg

get_ipython().magic('matplotlib inline')

once = False
def process_image(frame):
    global once
    if not once:
        cv2.imwrite("data/frame.jpg",frame)
        once = True
    return frame

img = mpimg.imread("data/frame.jpg")

fig = plt.figure("Frame", (10,10))
plt.imshow(img)
plt.show()

# Installing ffmpeg for playing video's in notebook
import imageio

imageio.plugins.ffmpeg.download()

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import os

INPUT_VIDEO  = os.path.join("data","lane_detection.mp4")

input_video = VideoFileClip(INPUT_VIDEO).subclip(0,30)
input_clip = input_video.fl_image(process_image)
#%time input_clip.write_videofile(OUTPUT_VIDEO, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(INPUT_VIDEO))



