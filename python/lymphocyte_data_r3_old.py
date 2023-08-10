import os
import sys
import numpy as np
import pandas as pd
import random
import glob
import scipy.misc
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
from PIL import Image
from numpy import linalg as LA
from numpy.random import choice
get_ipython().run_line_magic('matplotlib', 'inline')

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
    out_shape[2] = 2 # there are 2 classes

    delta=int((height)/2)
    image = np.lib.pad(image, ((delta, delta-int(downsample)), (delta, delta-int(downsample)), (0,0)), 'constant', constant_values=(0, 0))
    image = np.expand_dims(image, axis=0)
    heat = model.predict(image, batch_size=1, verbose=0)
    heat = np.reshape(heat, out_shape)
    # now apply the softmax to only the 3 classes(not the overall class probability (why??))
    heat[:,:,:] = np.apply_along_axis(softmax, 2, heat[:,:,:])
    return heat[:,:,1]

def extract_regions(data_loc, label_loc, out_dir, im_size, model, min_dist=3, samples_needed=400):
    # Num pos samples = 4 * actual number. (120ish)
    # sample the same number randomly, and from the hard class.(200 each)
    num_per_point=4
    elements = [-2, -1, 0, 1, 2] 
    weights = [0.1, 0.2, 0.4, 0.2, 0.1]
    
    im_size=int(im_size)

    all_locs = glob.glob(os.path.join(data_loc, '*'))
    folder_size = len(all_locs)
    print('folder_size: ', folder_size)
    
    all_labels=glob.glob(os.path.join(label_loc, '*'))
    all_labels = [loc for loc in all_labels if loc.rsplit('.', 1)[0][-1] == 'm']

    for image_file in all_locs:
        image = np.array(Image.open(image_file))[:,:,:3]
        
        # pad the image so you can always take the proper sized image
        delta=int((im_size)/2)+3
        image = np.lib.pad(image, ((delta, delta), (delta, delta), (0,0)), 'constant', constant_values=(0, 0))
        
        # get the labels. Find matching label image, and extract the coordinates:
        img_num = image_file.rsplit('/', 1)[1].rsplit('.',1)[0][2:]
        label_loc = next(loc for loc in all_labels if loc.rsplit('/', 1)[1].rsplit('.',1)[0][:-1]==img_num)

        label_img = np.asarray(Image.open(label_loc))
        label_list = np.asarray(np.where(label_img[:, :, 3]==255))
        label_list = np.transpose(label_list)
        label_list = label_list+delta

        # Get the positive samples:
        num_pos = 0
        for point in label_list:
            for i in range(num_per_point):
                y = point[0] + np.random.choice(elements, p=weights)
                x = point[1] + np.random.choice(elements, p=weights)
                seg_image = image[y-delta+3:y+delta-3, x-delta+3:x+delta-3,:]

                out_name=str(1)+'_'+str(num_pos)+'_'+img_num+'.jpg'
                outfile=os.path.join(out_dir, out_name)
                scipy.misc.imsave(outfile, seg_image)
                num_pos = num_pos+1

        # evenly sample the negatives from every image:
        samp_taken = 0
        while(samp_taken < samples_needed/2):
            row = random.randint(delta, image.shape[0]-delta)
            col = random.randint(delta, image.shape[1]-delta)
            proposed_center = np.array([row, col])
            dists = np.sqrt(np.sum((label_list - proposed_center) ** 2, axis=1))
            min_ind = np.argmin(dists)
            if (dists[min_ind] > min_dist+.5):
                seg_image = image[row-delta:row+delta, col-delta:col+delta,:]
                out_name=str(0)+'_'+str(samp_taken)+'_'+img_num+'.jpg'
                outfile=os.path.join(out_dir, out_name)
                scipy.misc.imsave(outfile, seg_image)
                samp_taken=samp_taken+1
                
        # Sample from the hard locations:
        heatmap = create_heatmap_L(image_file, model=model, height=im_size, downsample=2)
        # upsample to make everything easier
        heatmap = scipy.misc.imresize(heatmap, (100, 100))

        # Remove the true points
        heatmap = np.lib.pad(heatmap, ((3, 3), (3, 3)), 'constant', constant_values=(0, 0))
        label_list = label_list-delta + 3 # change it back to the actual points

        for point in label_list:
            for row in range(-1*min_dist, min_dist+1, 1):
                for col in range(-1*min_dist, min_dist+1, 1):
                    # dont't just do a square
                    dist = np.sqrt(row** 2 + col** 2)
                    if (dist<=min_dist+.5):
                        try:
                            # classifier predicted top left point, so by adding 1 this centers it a bit better
                            heatmap[int(point[0]+row+1), int(point[1]+col+1)] = 0 
                        except Exception:
                            continue
        heatmap = heatmap[3:-3, 3:-3]
                        
        # get the top samples_needed points, sample samples_needed/2 of them
        sample_point_list = np.unravel_index(np.argsort(heatmap.ravel())[-int(samples_needed):], heatmap.shape)
        sample_point_list = np.transpose(np.asarray(sample_point_list))
        idx = np.random.choice(len(sample_point_list), int(samples_needed/2))
        sample_point_list = sample_point_list[idx]

        for point in sample_point_list:
            y = point[0] + delta
            x = point[1] + delta
            seg_image = image[y-delta:y+delta, x-delta:x+delta,:]

            out_name=str(0)+'_'+str(num_pos)+'_'+img_num+'.jpg'
            outfile=os.path.join(out_dir, out_name)
            scipy.misc.imsave(outfile, seg_image)
            num_pos = num_pos+1

import keras
from keras.models import load_model, Model

import sys
sys.path.insert(0, '/Users/rb/Google_Drive/Waterloo/projects/cancer_hist/src')
from functions import*
from heat_models import*

in_dir='/Users/rb/Documents/waterloo/projects/cancer_hist/lymphocyte/ttv_split/'
label_loc='/Users/rb/Documents/waterloo/projects/cancer_hist/lymphocyte/manual_seg/center/'
out_dir='/Users/rb/Documents/waterloo/projects/cancer_hist/lymphocyte/extracted_new_32'

# model_loc='/Users/rb/Google_Drive/Waterloo/projects/cancer_hist/output/lymphocyte/conv_incp3_l/conv_incp3_64_.41-0.97.hdf5'
model_loc='/Users/rb/Google_Drive/Waterloo/projects/cancer_hist/output/lymphocyte/conv_incp3_l_32/conv_incp3_32_.20-0.96.hdf5'

model = conv_incp3(im_size=32)
model.load_weights(model_loc)

# First make the folders:
train_dir_in=os.path.join(in_dir, "train")
valid_dir_in=os.path.join(in_dir, 'valid')
test_dir_in=os.path.join(in_dir, 'test')

train_dir_out=os.path.join(out_dir, "train")
valid_dir_out=os.path.join(out_dir, 'valid')
test_dir_out=os.path.join(out_dir, 'test')

if not os.path.exists(train_dir_out):
    os.makedirs(train_dir_out)
if not os.path.exists(valid_dir_out):
    os.makedirs(valid_dir_out)
if not os.path.exists(test_dir_out):
    os.makedirs(test_dir_out)

if not os.path.exists(train_dir_in):
    os.makedirs(train_dir_in)
if not os.path.exists(valid_dir_in):
    os.makedirs(valid_dir_in)
if not os.path.exists(test_dir_in):
    os.makedirs(test_dir_in)

extract_regions(data_loc=test_dir_in, label_loc=label_loc, out_dir=test_dir_out, im_size=32, model=model, min_dist=2, samples_needed=400)
extract_regions(data_loc=train_dir_in, label_loc=label_loc, out_dir=train_dir_out, im_size=32, model=model, min_dist=2, samples_needed=400)
extract_regions(data_loc=valid_dir_in, label_loc=label_loc, out_dir=valid_dir_out, im_size=32, model=model, min_dist=2,  samples_needed=400)

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
    out_shape[2] = 2 # there are 2 classes

    delta=int((height)/2)
    image = np.lib.pad(image, ((delta, delta-int(downsample)), (delta, delta-int(downsample)), (0,0)), 'constant', constant_values=(0, 0))
    image = np.expand_dims(image, axis=0)
    heat = model.predict(image, batch_size=1, verbose=0)
    heat = np.reshape(heat, out_shape)
    # now apply the softmax to only the 3 classes(not the overall class probability (why??))
    heat[:,:,:] = np.apply_along_axis(softmax, 2, heat[:,:,:])
    return heat[:,:,1]


def extract_regions_n2(data_loc, label_loc, out_dir, im_size, model, min_dist=2, samples_needed=80):
    im_size=int(im_size)
    
    all_locs = glob.glob(os.path.join(data_loc, '*'))
    folder_size = len(all_locs)
    print('folder_size: ', folder_size)
    
    all_labels=glob.glob(os.path.join(label_loc, '*'))
    all_labels = [loc for loc in all_labels if loc.rsplit('.', 1)[0][-1] == 'm']

    for image_file in all_locs:
        image = np.array(Image.open(image_file))[:,:,:3]
        
        # pad the image so you can always take the proper sized image
        delta=int((im_size)/2)
        image = np.lib.pad(image, ((delta, delta), (delta, delta), (0,0)), 'constant', constant_values=(0, 0))
        
        # get the labels. Find matching label image, and extract the coordinates:
        img_num = image_file.rsplit('/', 1)[1].rsplit('.',1)[0][2:]
        label_loc = next(loc for loc in all_labels if loc.rsplit('/', 1)[1].rsplit('.',1)[0][:-1]==img_num)

        label_img = np.asarray(Image.open(label_loc))
        label_list = np.asarray(np.where(label_img[:, :, 3]==255))
        label_list = np.transpose(label_list)
        
        num_pos = 0
        for point in label_list:
            y = point[0] + delta
            x = point[1] + delta

            seg_image = image[y-delta:y+delta, x-delta:x+delta,:]
            out_name=str(1)+'_'+str(num_pos)+'_'+img_num+'.jpg'

            outfile=os.path.join(out_dir, out_name)
            scipy.misc.imsave(outfile, seg_image)
            num_pos = num_pos+1
    
        # evenly sample the negatives from every image:
        samp_taken = 0
        while (samp_taken < samples_needed):
            row = random.randint(delta, image.shape[0]-delta)
            col = random.randint(delta, image.shape[1]-delta)
            proposed_center = np.array([row, col])
            dists = np.sqrt(np.sum((label_list - proposed_center) ** 2, axis=1))
            min_ind = np.argmin(dists)
            if (dists[min_ind] > min_dist):
                seg_image = image[row-delta:row+delta, col-delta:col+delta,:]
                out_name=str(0)+'_'+str(samp_taken)+'_'+img_num+'.jpg'
                outfile=os.path.join(out_dir, out_name)
                scipy.misc.imsave(outfile, seg_image)
                samp_taken=samp_taken+1
                
        # Sample from the hard locations:
        heatmap = create_heatmap_L(image_file, model=model, height=im_size, downsample=2)
        # upsample to make everything easier
        heatmap = scipy.misc.imresize(heatmap, (100, 100))

        # Remove the true points
        heatmap = np.lib.pad(heatmap, ((3, 3), (3, 3)), 'constant', constant_values=(0, 0))
        for point in label_list:
            for row in range(-1*min_dist, min_dist+1, 1):
                for col in range(-1*min_dist, min_dist+1, 1):
                    # dont't just do a square
                    dist = np.sqrt(row** 2 + col** 2)
                    if (dist<=min_dist+.5):
                        try:
                            # classifier predicted top left point, so by adding 1 this centers it a bit better
                            heatmap[int(point[0]+row+1), int(point[1]+col+1)] = 0 
                        except Exception:
                            continue
        heatmap = heatmap[3:-3, 3:-3]
        
        # get the top samples_needed points, sample samples_needed/2 of them
        sample_point_list = np.unravel_index(np.argsort(heatmap.ravel())[-int(200):], heatmap.shape)
        sample_point_list = np.transpose(np.asarray(sample_point_list))
        idx = np.random.choice(len(sample_point_list), int(samples_needed/2))
        sample_point_list = sample_point_list[idx]

        for point in sample_point_list:
            y = point[0] + delta
            x = point[1] + delta
            seg_image = image[y-delta:y+delta, x-delta:x+delta,:]

            out_name=str(0)+'_'+str(num_pos)+'_'+img_num+'.jpg'
            outfile=os.path.join(out_dir, out_name)
            scipy.misc.imsave(outfile, seg_image)
            num_pos = num_pos+1

import keras
from keras.models import load_model, Model

import sys
sys.path.insert(0, '/Users/rb/Google_Drive/Waterloo/projects/cancer_hist/src')
from functions import*
from heat_models import*

in_dir='/Users/rb/Documents/waterloo/projects/cancer_hist/lymphocyte/ttv_split/'
label_loc='/Users/rb/Documents/waterloo/projects/cancer_hist/lymphocyte/manual_seg/center/'
out_dir='/Users/rb/Documents/waterloo/projects/cancer_hist/lymphocyte/extracted_new2_64_160'

model_loc='/Users/rb/Google_Drive/Waterloo/projects/cancer_hist/output/lymphocyte/conv_incp3_l/conv_incp3_64_.41-0.97.hdf5'
# model_loc='/Users/rb/Google_Drive/Waterloo/projects/cancer_hist/output/lymphocyte/conv_incp3_l_32/conv_incp3_32_.20-0.96.hdf5'

model = conv_incp3(im_size=64)
model.load_weights(model_loc)

# First make the folders:
train_dir_in=os.path.join(in_dir, "train")
valid_dir_in=os.path.join(in_dir, 'valid')
test_dir_in=os.path.join(in_dir, 'test')

train_dir_out=os.path.join(out_dir, "train")
valid_dir_out=os.path.join(out_dir, 'valid')
test_dir_out=os.path.join(out_dir, 'test')

if not os.path.exists(train_dir_out):
    os.makedirs(train_dir_out)
if not os.path.exists(valid_dir_out):
    os.makedirs(valid_dir_out)
if not os.path.exists(test_dir_out):
    os.makedirs(test_dir_out)

if not os.path.exists(train_dir_in):
    os.makedirs(train_dir_in)
if not os.path.exists(valid_dir_in):
    os.makedirs(valid_dir_in)
if not os.path.exists(test_dir_in):
    os.makedirs(test_dir_in)

extract_regions_n2(data_loc=test_dir_in, label_loc=label_loc, out_dir=test_dir_out, im_size=64, model=model, min_dist=2, samples_needed=160)
extract_regions_n2(data_loc=train_dir_in, label_loc=label_loc, out_dir=train_dir_out, im_size=64, model=model, min_dist=2, samples_needed=160)
extract_regions_n2(data_loc=valid_dir_in, label_loc=label_loc, out_dir=valid_dir_out, im_size=64, model=model, min_dist=2,  samples_needed=160)



