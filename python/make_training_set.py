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
train_folder = '/media/rene/Data/CAMELYON16/TrainingData'
base_out_dir = '/media/rene/Data/camelyon_out'

normal_locs = glob.glob(os.path.join(train_folder, 'Train_Normal/*'))
tumor_locs = glob.glob(os.path.join(train_folder, 'Train_Tumor/*'))
all_locs = normal_locs+tumor_locs

tile_num_list = []

for loc in all_locs:
    wsi = WSI(loc)
    wsi.generate_mask(mask_level=6)
    num_tiles = wsi.est_total_tiles(tile_size = 224)
    tile_num_list.append(num_tiles)
    
average_tiles = np.average(np.array(tile_num_list))
print('average_tiles: ', average_tiles)
print('max tiles: ', np.amax(np.array(tile_num_list)))
print('min tiles: ', np.amin(np.array(tile_num_list)))

np.random.seed(101)
vaild_normal_idx = np.random.choice(160, 32)
valid_tumor_idx = np.random.choice(110, 22)

average_tiles = 27098
tile_size = 224
base_out_dir = '/media/rene/Data/camelyon_out/basic_tiles_224'


for loc in all_locs[0:1]:
    if 'Normal' in loc:
        wsi_type = 'normal'
        wsi_id = int(loc.rsplit('_', 1)[-1].rsplit('.', 1)[0])
        if wsi_id in vaild_normal_idx:
            ttv = 'valid'
        else:
            ttv = 'train'

    elif 'Tumor' in loc:
        wsi_type = 'tumor'
        wsi_id = int(loc.rsplit('_', 1)[-1].rsplit('.', 1)[0])
        if wsi_id in valid_tumor_idx:
            ttv = 'valid'
        else:
            ttv = 'train'
    else:
        print('Error, not found as normal or tumor')
        
    # now read in and get the samples:
    wsi = WSI(loc)
    wsi.generate_mask(mask_level=6)
    total_tiles = wsi.est_total_tiles(tile_size = 224)
    num_tiles = np.amin([total_tiles, average_tiles/2])
    
    # Make folders for normal, tumor. Save each set of samples from a wsi in a folder within these.
    out_dir = os.path.join(base_out_dir, ttv, wsi_type, wsi.wsi_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Now make the tiles
    wsi.make_tiles(out_dir, num_tiles, tile_size)





