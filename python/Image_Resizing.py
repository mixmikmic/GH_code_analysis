import os
from PIL import Image
from __future__ import division

in_dir1 = "E:/erikn/Dropbox (DATA698-S17)/DATA698-S17/data/ddsm/png/0/"
in_dir2 = "E:/erikn/Dropbox (DATA698-S17)/DATA698-S17/data/ddsm/png/1/"
in_dir3 = "E:/erikn/Dropbox (DATA698-S17)/DATA698-S17/data/ddsm/png/3/"

img_in = [in_dir1, in_dir2, in_dir3]

out_dir = "E:/erikn/Documents/GitHub/MLProjects/data698_images/small/"

def basic_resize(height, width, in_dir, out_dir):
    # Takes a directory or list of directories containing png images and resizes to the given height and width 
    for directory in in_dir:
        images = os.listdir(directory)
        for img in images:
            im = Image.open(os.path.join(directory, img))
            size = im.resize((width,height), resample=Image.LANCZOS)
            size.save(os.path.join(out_dir, img))

height = 255
width = 255
basic_resize(height, width, img_in, out_dir)

from scipy import misc
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import time
get_ipython().magic('matplotlib inline')

# Finds the Maximum height and width from all of the images from the first run we have 
# Max Height = 7111
# Max Width = 5641
# Count = 4005
height = 0
width = 0
count = 0

start = time.time()
for directory in img_in:
    images = os.listdir(directory)
    for img in images:
        im = misc.imread(os.path.join(directory, img), flatten=False, mode='L')
        if im.shape[0] > height:
            height = im.shape[0]
        if im.shape[1] > width:
            width = im.shape[1]
        count += 1

print("Max Height, Max Width, Number of Images")
print(height, width, count)
end = time.time()
print("Time taken:")
print(end - start)

img_in = ["E:/erikn/Documents/GitHub/MLProjects/data698_images/png/"] # This is a test set use the img_in from above for all images
images = os.listdir(img_in[0])
im = Image.open(os.path.join(img_in[0], images[0]))
img_w, img_h = im.size
plt.imshow(im,cmap = plt.get_cmap('gray'))

background = Image.new('L', (5641, 7111), (0))
bg_w, bg_h = background.size
offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2) # Use // division in Python 3.5 
background.paste(im, offset)
plt.imshow(background,cmap = plt.get_cmap('gray'))

height = 400
width = int((height/img_h)*img_w)
print(height,width)
size = im.resize((width,height), resample=Image.LANCZOS)
background = Image.new('L', (height,height), (0))
bg_w, bg_h = background.size
offset = ((bg_w - width) // 2, (bg_h - height) // 2) # Use // division in Python 3.5 
background.paste(size, offset)
plt.imshow(background,cmap = plt.get_cmap('gray'))

def aspect_resize(height, in_dir, out_dir, square):
    for directory in in_dir:
        images = os.listdir(directory)
        for img in images:
            im = Image.open(os.path.join(directory, img))
            img_w, img_h = im.size
            if square == True:
                width = int((height/img_h)*img_w)
                size = im.resize((width,height), resample=Image.LANCZOS)
                background = Image.new('L', (height,height), (0))
                bg_w, bg_h = background.size
                offset = ((bg_w - width) // 2, (bg_h - height) // 2) # Use // division in Python 3.5 
                background.paste(size, offset)
            else:
                width = int((height/im.size[1])*im.size[0])
                background = im.resize((width,height), resample=Image.LANCZOS)
            background.save(os.path.join(out_dir, img))    

img_in = ["E:/erikn/Documents/GitHub/MLProjects/data698_images/png/"] # This is a test set use the img_in from above for all images
img_out = "E:/erikn/Documents/GitHub/MLProjects/data698_images/non_square/"
height = 150
aspect_resize(height,img_in, img_out, False)

img_in = ["E:/erikn/Documents/GitHub/MLProjects/data698_images/png/"] # This is a test set use the img_in from above for all images
img_out = "E:/erikn/Documents/GitHub/MLProjects/data698_images/square/"
height = 150
aspect_resize(height,img_in, img_out, True)

