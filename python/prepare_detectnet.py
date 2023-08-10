from PIL import Image
import os
import json

size = [720, 480]  # All input images would be resized accordingly
val_ratio = 0.2    # how many images would be reserved for validation

work_dir = './'

if not os.path.exists(work_dir + 'train_720x480/'):
        os.makedirs(work_dir + 'train_720x480/')

input_dir = work_dir + 'annotations/'
output_dir = work_dir + 'train_720x480/'
json_file = 'whale_faces_Vinh.json'

box_widths  = []  # widths of the boudning boxes after resizing
box_heights = []  # heights of the boudning boxes after resizing

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read json
with open(input_dir+json_file) as json_data:
    d = json.load(json_data)

for i in range(len(d)):
    # Resize image
    fn =  str(d[i]['filename'])    
    im = Image.open(os.path.join(input_dir,fn))
    print('{}: {}x{}'.format(fn,im.size[0],im.size[1]))
    # Pick the side which needs to be scaled down the most
    scale = min(float(size[0])/im.size[0],float(size[1])/im.size[1])
    im = im.resize((int(im.size[0]*scale),int(im.size[1]*scale)),
                   Image.ANTIALIAS)
    # Padding on either right or bottom if necessary
    im = im.crop((0,0,size[0],size[1]))
    resized_path = output_dir + os.path.basename(fn)
    # Save the resized image
    im.save(resized_path)
    
    # Modify bounding boxes to match image scaling
    for j in range(len(d[i]['annotations'])):
        x = d[i]['annotations'][j]['x'] * scale
        y = d[i]['annotations'][j]['y'] * scale
        w = d[i]['annotations'][j]['width'] * scale
        h = d[i]['annotations'][j]['height'] * scale
        # Fix to avoid x, y, w, h out of bound
        if x < 0:
            w += x
            if w < 0: w = 20  # this should not happen
            x = 0
        if y < 0:
            h += y
            if h < 0: h = 20  # this should not happen
            y = 0
        if x+w > size[0]:
            w = size[0] - x
        if y+h > size[1]:
            h = size[1] - y
        d[i]['annotations'][j]['x'] = x
        d[i]['annotations'][j]['y'] = y
        d[i]['annotations'][j]['width'] = w 
        d[i]['annotations'][j]['height'] = h
        d[i]['filename'] = '../train_720x480/' + os.path.basename(fn)
        box_widths.append(w)
        box_heights.append(h)

# Save the updated JSON file
with open(input_dir + 'whale_faces_720x480.json', 'w') as fp:
    json.dump(d, fp, indent=0)

import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print('*** after resizing ***')
print('min/max bounding box widths  = {0:4.1f} / {1:4.1f}'.format(
    max(box_widths),min(box_widths)))
print('min/max bounding box heights = {0:4.1f} / {1:4.1f}'.format(
    max(box_heights),min(box_heights)))

plt.figure(figsize=(5, 5))

plt.subplot(1, 1, 1)
plt.scatter(np.array(box_widths),np.array(box_heights))
plt.title('bounding box widths/heights')
plt.xlabel('width')
plt.ylabel('height')

plt.show()

import random
import shutil

if os.path.exists(work_dir + 'detectnet_720x480/'):
    shutil.rmtree(work_dir + 'detectnet_720x480/')
os.makedirs(work_dir + 'detectnet_720x480/train/images/')
os.makedirs(work_dir + 'detectnet_720x480/train/labels/')
os.makedirs(work_dir + 'detectnet_720x480/val/images/')
os.makedirs(work_dir + 'detectnet_720x480/val/labels/')

input_dir = work_dir + 'train_720x480/'
json_file = './annotations/whale_faces_720x480.json'

# Read json
with open(json_file) as json_data:
    d = json.load(json_data)

for i in range(len(d)):
    output_dir = work_dir + 'detectnet_720x480/train/'
    if random.random() < val_ratio:
        output_dir = work_dir + 'detectnet_720x480/val/'
    # Copy the image over
    fn =  str(d[i]['filename'])
    shutil.copy(input_dir + fn, output_dir + 'images/')
    bn = os.path.basename(fn)
    fnbase, ext = os.path.splitext(bn)
    # One Label file per one image
    with open(output_dir + 'labels/' + fnbase + '.txt', 'w') as fp:
        # Convert annotations to required format
        for j in range(len(d[i]['annotations'])):
            l = d[i]['annotations'][j]['x']
            t = d[i]['annotations'][j]['y']
            r = l + d[i]['annotations'][j]['width']
            b = t + d[i]['annotations'][j]['height']
            
            type = 'Car'
            truncated = 0
            occluded  = 3
            alpha  = 0
            tail = '0 0 0 0 0 0 0 0'
            
            label = type + ' ' +                                str(truncated) + ' ' +                      str(occluded)  + ' ' +                      str(alpha)     + ' ' +                      str(l) + ' ' + str(t) + ' ' + str(r) + ' ' + str(b) + ' ' + tail
            fp.write(label + '\n')

get_ipython().system('find detectnet_720x480/train/ -name "*.jpg" | wc -l')
get_ipython().system('find detectnet_720x480/train/ -name "*.txt" | wc -l')
get_ipython().system('find detectnet_720x480/val/ -name "*.jpg" | wc -l')
get_ipython().system('find detectnet_720x480/val/ -name "*.txt" | wc -l')



