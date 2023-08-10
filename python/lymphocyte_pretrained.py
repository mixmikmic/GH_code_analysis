import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
from PIL import Image
import keras
from keras.models import load_model, Model

# model_loc = '/Users/rb/Google_Drive/Waterloo/projects/cancer_hist/output/conv_incp3_py3/conv_incp3_128_.46-0.91.hdf5'
model_loc = '/Users/rb/Google_Drive/Waterloo/projects/cancer_hist/output/conv_incp3/conv_incp3_64_.156-0.92.hdf5'
model = load_model(model_loc)

get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.insert(0, '/Users/rb/Google_Drive/Waterloo/projects/cancer_hist/src')
from functions import*

def create_heatmap_L(image_loc, model, height, downsample):
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    image = np.asarray(Image.open(image_loc))[:,:,:3]
    image_shape = image.shape
    image = image/255.0 # During training the images were normalized
    height = int(height)

    last = model.layers[-2].output
    model = Model(model.input, last)

    out_shape = np.ceil(np.array(image.shape)/float(downsample)).astype(int)
    out_shape[2] = 4 # there are 4 classes

    delta=int((height)/2)
    image = np.lib.pad(image, ((delta, delta-int(downsample)), (delta, delta-int(downsample)), (0,0)), 'constant', constant_values=(0, 0))
    image = np.expand_dims(image, axis=0)
    heat = model.predict(image, batch_size=1, verbose=0)
    heat = np.reshape(heat, out_shape)
    
    # now apply the softmax to only the 3 classes(not the overall class probability (why??))
    heat[:,:,:] = np.apply_along_axis(softmax, 2, heat[:,:,:])
    
    f = plt.figure(figsize=(6, 6))
    plt.imshow(heat[:,:,0])
    f = plt.figure(figsize=(6, 6))
    plt.imshow(heat[:,:,1])
    f = plt.figure(figsize=(6, 6))
    plt.imshow(heat[:,:,2])
    f = plt.figure(figsize=(6, 6))
    plt.imshow(heat[:,:,3])
    return heat[:,:,0]


def non_max_supression(heatmap, radius=5, cutoff=.5, stride=2): 
    labels = []
    points = []    
    heatmap = np.lib.pad(heatmap, ((radius, radius), (radius, radius)), 'constant', constant_values=(0, 0))

    curr_max = 1
    while (curr_max > float(cutoff)):
        max_coord = np.asarray(np.unravel_index(heatmap[:, :].argmax(), heatmap[:, :].shape))
            
        # find the max set all classes within radius r to p = 0
        curr_max = heatmap[:, :].max()
        for row in range(-1*radius, radius, 1):
            for col in range(-1*radius, radius, 1):
                # dont't just do a square
                dist = np.sqrt(row** 2 + col** 2)
                if (dist<=radius):
                    heatmap[int(max_coord[0]+row), int(max_coord[1]+col)] = 0
        # adjust for the padding that was added
        max_coord[0] = max_coord[0] - radius
        max_coord[1] = max_coord[1] - radius
        
        points.append(max_coord)
    points = np.array(points)
    #points = points[points[:,2]!=0]
    
    points = points.astype(float)
    points[:,0] = points[:,0]*stride
    points[:,1] = points[:,1]*stride
    return points

loc = '/Users/rb/Documents/waterloo/projects/cancer_hist/lymphocyte/ttv_split/test'
all_imgs=glob.glob(os.path.join(loc, '*'))

label_locs = '/Users/rb/Documents/waterloo/projects/cancer_hist/lymphocyte/manual_seg/center/'
all_labels=glob.glob(os.path.join(label_locs, '*'))
all_labels = [loc for loc in all_labels if loc.rsplit('.', 1)[0][-1] == 'm']

cutoff = .02
radius = 4
cor_rad = 8

all_matched_pts = []
all_matched_preds = []
abs_error_list = []
total_nuclei = 0
num_predicted = 0

for image_loc in all_imgs:
    # get the labels. Find matching label image, and extract the coordinates:
    img_num = image_loc.rsplit('/', 1)[1].rsplit('.',1)[0][2:]
    label_loc = next(loc for loc in all_labels if loc.rsplit('/', 1)[1].rsplit('.',1)[0][:-1]==img_num)
    label_img = np.asarray(Image.open(label_loc))
    label_list = np.asarray(np.where(label_img[:, :, 3]==255))
    label_list = np.transpose(label_list)
    
    heatmap = create_heatmap_L(image_loc, model=model, height=64, downsample=2)
    preds = non_max_supression(heatmap, radius=radius, cutoff=cutoff, stride=2)
    acc = get_matched_pts2(label_list, preds, cor_rad=cor_rad)
    
    print('len(acc[all_matched_preds]), acc[total_nuclei]', len(acc["all_matched_preds"]), acc['total_nuclei'])
    
    # Update 
    all_matched_preds.extend(np.array(acc["all_matched_preds"]))
    all_matched_pts.extend(np.array(acc["all_matched_pts"]))
    total_nuclei = total_nuclei + acc['total_nuclei']
    num_predicted = num_predicted + acc['num_predicted']
    abs_error_list.append(acc['abs_error'])
    
print '% pred, not matched / total true: ', (float(num_predicted)-len(all_matched_pts))/float(total_nuclei)

print acc["total_nuclei"]
print acc["num_predicted"]

image_loc = '/Users/rb/Documents/waterloo/projects/cancer_hist/lymphocyte/ttv_split/train/im8.tif'

img_num = image_loc.rsplit('/', 1)[1].rsplit('.',1)[0][2:]
label_loc = next(loc for loc in all_labels if loc.rsplit('/', 1)[1].rsplit('.',1)[0][:-1]==img_num)
label = np.asarray(Image.open(label_loc))
label_list = np.asarray(np.where(label[:, :, 3]==255))
true_pts = np.transpose(label_list)

heat = create_heatmap_L(image_loc, model=model, height=64, downsample=2)
points = non_max_supression(heat, radius=5, cutoff=.05, stride=2)
acc = get_matched_pts2(true_pts, preds, cor_rad=5)

image = np.asarray(Image.open(image_loc))[:,:,:3]
print 'image.shape', image.shape
print 'len(true_pts)', len(true_pts)
print 'len(points)', len(points)
print 'acc["num_predicted"]', acc["num_predicted"]
print 'acc["total_nuclei"]', acc["total_nuclei"]

image = np.asarray(Image.open(image_loc))[:,:,:3]
image.setflags(write=1)
print(true_pts)
f = plt.figure(figsize=(17, 9))
for row in range(len(true_pts)):
    color = [0, 255, 0]
    image[int(true_pts[row, 0])-2:int(true_pts[row, 0])+2, int(true_pts[row, 1])-2:int(true_pts[row, 1])+2, :] = color
plt.imshow(image)

