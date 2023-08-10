import sys
import subprocess

from IPython.display import YouTubeVideo

# Full Screen Video
videoId = "xPJCiML28Ds"
start = 2
duration = 10

# Obtain yt link
ytCmd = ["youtube-dl","-f", "best", "--get-url", "https://www.youtube.com/watch?v="+videoId]
p = sp.Popen(ytCmd, stdin=sp.PIPE, stdout=sp.PIPE)
videoURL = p.communicate()[0].strip()
print videoURL

# Obtain yt link
ytCmd = ["youtube-dl","-f", "best", "--get-url", "https://www.youtube.com/watch?v="+videoId]
p = sp.Popen(ytCmd, stdin=sp.PIPE, stdout=sp.PIPE)
videoURL = p.communicate()[0].strip()
print videoURL

get_ipython().run_cell_magic('bash', "-s '$outputPath' '$modifiedPath' ", "# temporarily hardcoding crop of black bars, should be auto/user prevented.  4:3 on crt's\n $2 ")

get_ipython().run_cell_magic('bash', "-s '$modifiedPath' '$percentPath' ", '# Take 4:3 or w/e melee video and capture text reigon\n#ffmpeg -y -i $1 -vf crop=iw-20:30:10:ih-55,crop=iw/2:ih:0:0,edgedetect=low=0.1:high=0.4 $2\nffmpeg -y -i $1 -vf crop=iw/6:40:20:ih-60 $2')

from IPython.display import HTML
HTML("""
<video>
  <source src="percents.mp4" type="video/mp4">
</video>
""")

get_ipython().run_cell_magic('bash', '', "#ffmpeg -y -ss 2.5 -i percents.mp4 select='gt(scene\\,0.9)' -vsync 0 -an keyframes%03d.jpg\n\nffmpeg -y -ss 6.7 -i percents.mp4 -vframes 1 -q:v 2 percent.jpg\n\nffmpeg -y -ss 8 -i percents.mp4 -vframes 1 -q:v 2 percent2.jpg")

get_ipython().run_cell_magic('bash', '', 'tesseract percent.jpg stdout -l eng digits')

import cv2
import numpy as np
import subprocess
from PIL import Image#, ImageOps, ImageFilter
def binarize(image):
    return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]

# Find the active chord for each frame
def vidcap_to_frame_chords(vidcap, video_fps, nb_frames = -1):
    if nb_frames == -1:
        nb_frames = int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    frame_chords = []
    final_frame = int(nb_frames - REMOVE_FINAL * video_fps)
    for i in range(final_frame - 1):
        success,image = vidcap.read()
        frame_chords.append(get_active_chord(image))
    return frame_chords

vidcap = cv2.VideoCapture("percents.mp4")
video_fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
video_nb_frames = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

IMG_FILENAME = "ocr.png"
frame_percents = []
for i in range(int(video_nb_frames)):
    success,image = vidcap.read()
    image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3,3),np.uint8)
    image = cv2.erode(image,kernel)
    
    # Save image
    Image.fromarray(image).save(IMG_FILENAME)
    output = subprocess.check_output("tesseract "+IMG_FILENAME+" stdout -l eng -psm 8 digits", shell=1)
    frame_percents.append(output)
    print output
    

frame_percents

get_ipython().run_cell_magic('latex', '', '')



