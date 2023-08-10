# Imports
import matplotlib.pyplot as plt
import pandas as pd
from skimage.color import rgb2grey
import moviepy.editor as mpy
from skimage.feature import corner_harris, corner_subpix, corner_peaks
import naive_solution as motd

import ipywidgets
from IPython.display import display

get_ipython().magic('matplotlib inline')

# load a video
motd_clip = motd.load_video('motd-sample.mp4')

# set up interactive slider for time of sample frame (seconds)
sample_time = 9*60 + 10  # 9 mins 10 secs

w = ipywidgets.IntSlider()
w.max = motd_clip.duration
w.value = sample_time
display(w)

# Get a frame with a match
sample_frame = motd_clip.get_frame(w.value)
bw_frame = rgb2grey(sample_frame)  # make bw for edge detection

# Find edges with skimage
coords = corner_peaks(corner_harris(bw_frame), min_distance=5)

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(sample_frame)
ax.scatter(coords[:, 1], coords[:, 0], alpha=0.5, color='skyblue', s=250)

