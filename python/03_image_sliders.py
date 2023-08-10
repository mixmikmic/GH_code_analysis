get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")
import pycamhd.pycamhd as camhd
import numpy as np
import matplotlib.pyplot as plt

filenames = []
with open('d3.txt') as f:
    for line in f:
        filenames.append(line)

def show_image(file_number, frame_number):
    filename = filenames[file_number]
    plt.rc('figure', figsize=(10, 5))
    plt.rcParams.update({'font.size': 8})
    frame = camhd.get_frame(filename, frame_number, 'rgb24')
    fig, ax = plt.subplots();
    im1 = ax.imshow(frame);
    plt.yticks(np.arange(0,1081,270))
    plt.xticks(np.arange(0,1921,480))
    plt.title('%s (%i:%i)' % (filename[84:].strip(), file_number, frame_number));

from ipywidgets import interact
from ipywidgets import IntSlider
file_slider = IntSlider(min=0, max=len(filenames)-1, step=1, value=0, continuous_update=False)
frame_slider = IntSlider(min=0, max=camhd.get_frame_count(filenames[0]), step=10, value=0, continuous_update=False)

def update_range(*args):
    frame_slider.max = camhd.get_frame_count(filenames[file_slider.value])-1
    if frame_slider.value > frame_slider.max:
        frame_slider.value = frame_slider.max
file_slider.observe(update_range, 'value')

interact(show_image, file_number=file_slider, frame_number=frame_slider);

