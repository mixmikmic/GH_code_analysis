from PIL import Image
import os
import json

work_dir = './'
# os.chdir(work_dir)

im_widths   = []  # widths of the original images
im_heights  = []  # heights of the original images
box_widths  = []  # widths of the boudning boxes (as specified in the JSON files)
box_heights = []  # heights of the boudning boxes (as specified in the JSON files)

input_dir = work_dir + 'annotations/'
json_file = 'whale_faces_Vinh.json'

# Read json
with open(input_dir+json_file) as json_data:
    d = json.load(json_data)
    
for i in range(len(d)):
    # Record image sizes
    fn =  str(d[i]['filename'])    
    im = Image.open(os.path.join(input_dir,fn))
    print('{}: {}x{}'.format(fn,im.size[0],im.size[1]))
    im_widths.append(im.size[0])
    im_heights.append(im.size[1])
    
    # Also record all bounding box widths and heights
    for j in range(len(d[i]['annotations'])):
        box_widths.append(d[i]['annotations'][j]['width'])
        box_heights.append(d[i]['annotations'][j]['height'])

import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.hist(np.array(im_widths))
plt.title('image widths')
plt.subplot(2, 2, 2)
plt.hist(np.array(im_heights))
plt.title('image heights')
plt.subplot(2, 2, 3)
plt.hist(np.array(box_widths))
plt.title('box widths')
plt.subplot(2, 2, 4)
plt.hist(np.array(box_heights))
plt.title('box heights')
plt.tight_layout()
plt.show()

print('min/max image widths  = {0:4d} / {1:4d}'.format(
    max(im_widths),min(im_widths)))
print('min/max image heights = {0:4d} / {1:4d}'.format(
    max(im_heights),min(im_heights)))

plt.figure(figsize=(5, 5))

plt.subplot(1, 1, 1)
plt.scatter(np.array(im_widths),np.array(im_heights))
plt.title('image widths/heights')
plt.xlabel('width')
plt.ylabel('height')

plt.show()

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



