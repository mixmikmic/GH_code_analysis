import os
import sys
import glob
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from skimage import filters
from skimage.morphology import disk
from openslide import OpenSlide, OpenSlideUnsupportedFormatError

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from IPython.display import display, HTML
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.sampler as sampler

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

# Add the src directory for functions
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'src')
print(src_dir)
sys.path.append(src_dir)

# import my functions:
from WSI_utils import*

# Base Directory where data is stored
base_data_dir = '/media/rene/Data/CAMELYON16'
base_out_dir = '/media/rene/Data/camelyon_out'

nontumor_data_dir = '/media/rene/Data/CAMELYON16/TrainingData/Train_Normal'
tumor_data_dir = '/media/rene/Data/CAMELYON16/TrainingData/Train_Tumor'

wsi_path = '/media/rene/Data/CAMELYON16/TrainingData/Train_Normal/Normal_001.tif'
wsi = OpenSlide(wsi_path)
print(wsi.level_count)

print('wsi.level_dimensions', wsi.level_dimensions)
print('wsi.level_downsamples', wsi.level_downsamples)

# read in the image at a low resolution (6) and convert to hsv
# is the 4 channel to 3-channel hsv conversion ok? Img looks resonable
img = wsi.read_region(location=(0, 0), level=6, size=wsi.level_dimensions[6]).convert('RGB')
plt.imshow(img)
print(type(img))
img = img.convert('HSV')
print(np.array(img).shape)
plt.figure()
plt.imshow(img)

wsi_path = '/media/rene/Data/CAMELYON16/TrainingData/Train_Normal/Normal_009.tif'
wsi = OpenSlide(wsi_path)

img = wsi.read_region(location=(0, 0), level=5, size=wsi.level_dimensions[5]).convert('RGB')
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(img)
fig.savefig('/media/rene/Data/camelyon_out/figures/wsi_example.jpeg', bbox_inches='tight', pad_inches=0)

# do otsu's method on the saturation ??? paper does it on hue and saturation, how???
img_saturation = np.asarray(img)[:, :, 1]
threshold = filters.threshold_otsu(img_saturation)
high_saturation = (img_saturation > threshold)
# otsu's on the hue:
img_hue = np.asarray(img)[:, :, 0]
threshold = filters.threshold_otsu(img_hue)
high_hue = (img_hue > threshold)

mask = high_saturation | high_hue

plt.figure()
plt.imshow(high_saturation)
plt.figure()
plt.imshow(high_hue)
plt.figure()
plt.imshow(mask)

def generate_mask(wsi, mask_level=8, disk_radius=False):
    img = wsi.read_region(location=(0, 0), level=mask_level, size=wsi.level_dimensions[mask_level]).convert('HSV')
    img_saturation = np.asarray(img)[:, :, 1]
    threshold = filters.threshold_otsu(img_saturation)
    high_saturation = (img_saturation > threshold)
    
    # optionally use the disk method (sensible radius was 10)
    if disk_radius!=False:
        disk_object = disk(disk_radius)
        mask = closing(high_saturation, disk_object)
        mask = opening(mask, disk_object)
    else: 
        mask = high_saturation
    return mask

mask = generate_mask(wsi, mask_level=7)
print(mask.shape)
plt.imshow(high_saturation)

def est_total_tiles(wsi, mask, mask_level):
    image_size = 224
    # NOT CORRECT WAY. Just estimate by comparing number of pixels in mask to size of img.
    # Should still give decent results though
    # a patch is the area of one pixel in the downsampled mask. Will be a square of length downsample factor
    num_patches = np.sum(mask)
    total_pixels = num_patches*wsi.level_downsamples[mask_level]**2
    total_tiles = total_pixels/image_size**2
    return total_tiles
    
def sample_tiles(wsi, wsi_name, mask, mask_level, out_dir, num_samples, tile_size=224):        
    patch_size = np.round(mask_level) # size of each pixel in mask in terms of level 0
    curr_samples = 0
    locations = []
    while(curr_samples < num_samples):
        # randomly select a pixel in the mask to sample from
        all_indices = np.asarray(np.where(mask))
        idx = np.random.randint(0, len(all_indices[0]))
        sample_patch_ind = np.array([all_indices[1][idx], all_indices[0][idx]]) # not sure why this is backwards like that
        locations.append(sample_patch_ind)
        # convert to coordinates of level 0
        sample_patch_ind = np.round(sample_patch_ind*wsi.level_downsamples[mask_level])
        # random point inside this patch for center of new image (sampled image could extend outside patch or mask)
        # sample_index should be top left corner
        location = (np.random.randint(sample_patch_ind[0]-tile_size/2, sample_patch_ind[0]+tile_size/2),
                    np.random.randint(sample_patch_ind[1]-tile_size/2, sample_patch_ind[1]+tile_size/2))
        
        try:
            img = wsi.read_region(location=location, level=0, size=(tile_size, tile_size))
        except:
            continue # if exception try sampling a new location. Hopefully was was out of bounds
        curr_samples+=1
        
        img = img-np.amin(img) # left shift to 0
        img = (img/np.amax(img))*255 # scale to [0, 255]
        img = Image.fromarray(img.astype('uint8'))  

        out_file = os.path.join(out_dir, wsi_name +'_'+ str(curr_samples))

        img.save(out_file, 'PNG')
        plt.figure()
        plt.imshow(img)
    return locations

wsi_path = '/media/rene/Data/CAMELYON16/TrainingData/Train_Normal/Normal_001.tif'
wsi = OpenSlide(wsi_path)
wsi_name = 'Normal_001'
mask_level = 6
out_dir = '/media/rene/Data/camelyon_out/test'
num_samples = 5
 
mask = generate_mask(wsi, mask_level)
total_samples = est_total_tiles(wsi, mask, mask_level)
print(total_samples)
locations = sample_tiles(wsi, wsi_name, mask, mask_level, out_dir, num_samples, tile_size=224)

img = np.asarray(wsi.read_region(location=(0, 0), level=6, size=wsi.level_dimensions[6]).convert('RGB'))
plt.figure()
plt.imshow(mask)

# set the locations to a value
img.setflags(write=1)
for location in locations:
    img[location[1]-10:location[1]+10, location[0]-10:location[0]+10, :] = [0, 255, 0]
#     img[location[0], location[1],:] = [0, 255, 0]
        
plt.figure(figsize=(8, 8))
plt.imshow(img)

number = '011'
wsi = OpenSlide('/media/rene/Data/CAMELYON16/TrainingData/Train_Tumor/Tumor_'+number+'.tif')
img = wsi.read_region(location=(0, 0), level=6, size=wsi.level_dimensions[6]).convert('RGB')
plt.figure(figsize=(6, 6))
plt.imshow(img)

img = Image.open('/media/rene/Data/CAMELYON16/EvaluationMasks/Tumor_'+number+'_EvaluationMask.png')
mask = np.asarray(img)
print(mask.shape)
plt.figure(figsize=(6, 6))
plt.imshow(mask)

unique, counts = np.unique(mask, return_counts=True)
print(np.asarray((unique, counts)).T)

number = '011'
wsi = OpenSlide('/media/rene/Data/CAMELYON16/TrainingData/Train_Tumor/Tumor_'+number+'.tif')
img = wsi.read_region(location=(0, 0), level=6, size=wsi.level_dimensions[6]).convert('RGB')
plt.figure(figsize=(6, 6))
plt.imshow(img)
print(wsi.level_count)
print('wsi.level_dimensions', wsi.level_dimensions)
print('wsi.level_downsamples', wsi.level_downsamples)

wsi = OpenSlide('/media/rene/Data/CAMELYON16/TrainingData/Ground_Truth/Mask/Tumor_'+number+'_Mask.tif')
img = wsi.read_region(location=(0, 0), level=6, size=wsi.level_dimensions[6]).convert('RGB')
plt.figure(figsize=(6, 6))
plt.imshow(img)
print(wsi.level_count)
print('wsi.level_dimensions', wsi.level_dimensions)
print('wsi.level_downsamples', wsi.level_downsamples)

wsi_path = '/media/rene/Data/CAMELYON16/TrainingData/Train_Tumor/Tumor_'+number+'.tif'

annotation_path = '/media/rene/Data/CAMELYON16/TrainingData/Ground_Truth/Mask/Tumor_'+number+'_Mask.tif'

base_dir = wsi_path.rsplit('TrainingData', 1)[:-1][0]
annotation_file_name = wsi_path.rsplit('/', 1)[-1].replace(".tif", "_Mask.tif")
annotation_path = os.path.join(base_dir, 'TrainingData', 'Ground_Truth', 'Mask', annotation_file_name)
print(annotation_path)

tumor_mask_level = 5
tile_size=224
tile_sample_level=0

tumor_annotation_wsi = OpenSlide('/media/rene/Data/CAMELYON16/TrainingData/Ground_Truth/Mask/Tumor_001_Mask.tif')
tumor_mask = tumor_annotation_wsi.read_region(location=(0, 0), level=tumor_mask_level, 
    size=tumor_annotation_wsi.level_dimensions[tumor_mask_level]).convert('RGB')

unique, counts = np.unique(tumor_mask, return_counts=True)
print(np.asarray((unique, counts)).T)

wsi = WSI('/media/rene/Data/CAMELYON16/TrainingData/Train_Tumor/Tumor_011.tif')
out_dir = '/media/rene/Data/camelyon_out/test/normal'
num_tiles = 11
tile_class = 'tumor'

wsi.sample_from_tumor_region(out_dir, num_tiles, tile_size=224, tile_sample_level=0)

wsi = WSI('/media/rene/Data/CAMELYON16/TrainingData/Train_Tumor/Tumor_011.tif')
out_dir = '/media/rene/Data/camelyon_out/test/normal'
num_tiles = 11
tile_class = 'tumor'

wsi.sample_from_normal_region(out_dir, num_tiles, tile_size=224, tile_sample_level=0)

image_loc = '/media/rene/Data/camelyon_out/test/normal/Tumor_011_normal_4'

img=np.asarray(Image.open(image_loc))
print(img.shape)
plt.figure(figsize=(6, 6))
plt.imshow(img)

num_tiles = 11
out_dir = '/media/rene/Data/camelyon_out/test'

wsi = WSI('/media/rene/Data/CAMELYON16/TrainingData/Train_Tumor/Tumor_011.tif')
wsi.make_tiles_by_class(out_dir, num_tiles, tile_class='normal', tile_size=224, tile_sample_level=0)
wsi.make_tiles_by_class(out_dir, num_tiles, tile_class='tumor', tile_size=224, tile_sample_level=0)

wsi = WSI('/media/rene/Data/CAMELYON16/TrainingData/Train_Normal/Normal_001.tif')
wsi.make_tiles_by_class(out_dir, num_tiles, tile_class='normal', tile_size=224, tile_sample_level=0)
wsi.make_tiles_by_class(out_dir, num_tiles, tile_class='tumor', tile_size=224, tile_sample_level=0)

SEED = 101
np.random.seed(SEED)

data_loc = '/media/rene/Data/CAMELYON16'
out_file = '/media/rene/Data/camelyon/other/ttv_split.p'
valid_frac = .2

# args.data_loc is location of CAMELYON16 directory
normal_loc = os.path.join(data_loc, 'TrainingData', 'Train_Normal')
tumor_loc = os.path.join(data_loc, 'TrainingData', 'Train_Tumor')
num_normal = len(glob.glob(os.path.join(normal_loc, '*')))
num_tumor = len(glob.glob(os.path.join(tumor_loc, '*')))
num_all = num_normal+num_tumor

# create validation set. Randomly sample args.valid_frac of tumor and non-tumor training set
normal_wsi_locs = glob.glob(os.path.join(normal_loc, '*'))
normal_vaild_idx = np.random.choice(num_normal, int(np.round(num_normal*valid_frac)))
tumor_wsi_locs = glob.glob(os.path.join(tumor_loc, '*'))
tumor_vaild_idx = np.random.choice(num_tumor, int(np.round(num_tumor*valid_frac)))

ttv_split = {}
ttv_split['normal_vaild_idx'] = normal_vaild_idx
ttv_split['tumor_vaild_idx'] = tumor_vaild_idx

pickle.dump( ttv_split, open( out_file, "wb" ) )

split = pickle.load( open( out_file, "rb" ) )
print(split)

wsi_path = '/media/rene/Data/CAMELYON16/Testset/Ground_Truth/Masks/Test_116_Mask.tif'
wsi = OpenSlide(wsi_path)
print(wsi.level_count)

img = wsi.read_region(location=(0, 0), level=3, size=wsi.level_dimensions[3]).convert('RGB')
plt.figure(figsize=(12, 12))
plt.imshow(img)



