import dicom
import skimage

df = dicom.read_file('../images/lung.dcm')
image = skimage.img_as_int(df.pixel_array)

import sys
sys.path.append('..')
from livewire import LiveWireSegmentation

algorithm = LiveWireSegmentation(image,smooth_image=False, threshold_gradient_image=False)

import numpy as np

get_ipython().magic('matplotlib')
import matplotlib.pyplot as plt
plt.gray()

INTERACTIVE = True  # to compute interactive shortest path suggestions

from itertools import cycle
COLORS = cycle('rgbyc')  # use separate colors for consecutive segmentations

start_point = None
current_color = COLORS.next()
current_path = None
length_penalty = 10.0

def button_pressed(event):
    global start_point
    if start_point is None:
        start_point = (int(event.ydata), int(event.xdata))
        
    else:
        end_point = (int(event.ydata), int(event.xdata))
        
        # the line below is calling the segmentation algorithm
        path = algorithm.compute_shortest_path(start_point, end_point, length_penalty=length_penalty)
        plt.plot(np.array(path)[:,1], np.array(path)[:,0], c=current_color)
        start_point = end_point

def mouse_moved(event):
    if start_point is None:
        return
    
    end_point = (int(event.ydata), int(event.xdata))
    
    # the line below is calling the segmentation algorithm
    path = algorithm.compute_shortest_path(start_point, end_point, length_penalty=length_penalty)
    
    global current_path
    if current_path is not None:
        current_path.pop(0).remove()
    current_path = plt.plot(np.array(path)[:,1], np.array(path)[:,0], c=current_color)

def key_pressed(event):
    if event.key == 'escape':
        global start_point, current_color
        start_point = None
        current_color = COLORS.next()

        global current_path
        if current_path is not None:
            current_path.pop(0).remove()
            current_path = None
            plt.draw()

plt.connect('button_release_event', button_pressed)
if INTERACTIVE:
    plt.connect('motion_notify_event', mouse_moved)
plt.connect('key_press_event', key_pressed)

plt.imshow(image)
plt.autoscale(False)
plt.title('Livewire example')
plt.show()

