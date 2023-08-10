import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys, os, time, gc
import requests, shutil, random
from skimage import io
from skimage.transform import resize
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')

# Load the data
train = pd.read_csv('./data/all/train.csv')
test = pd.read_csv('./data/all/test.csv')

print('Train:\t\t', train.shape)
print('Test:\t\t', test.shape)

print('Landmarks:\t', len(train['landmark_id'].unique()))

train.head()

def valid(path):
    """ function to determine whether or not the given image is valid """
    try:
        img = Image.open(path)
        if img.width < 256 or img.height < 256 or img.format != 'JPEG':
            return False
        _ = img.resize((256, 256))
    except:
        return False
    
    return True

# Choose the unique ids
unique_ids = sorted(train['landmark_id'].unique())
len(unique_ids)

# fix random state
np.random.seed(42)
random.seed(29)

# Split into training and test set
train_idx = []
val_idx = []
test_idx = []

for landmark_id in unique_ids:
    # help information
    if landmark_id % 3000 == 0:
        print('\nProcess: {:8d}'.format(landmark_id))
    if landmark_id % 40 == 0:
        print('=', end='')
        
    # get index corresponding to given landmark_id
    index = list(train[train['landmark_id'] == landmark_id].index)
    np.random.shuffle(index)
    
    # check valid image numbers
    valid_idx = []
    for idx in index:
        path = './data/all/train_images/' + str(idx) + '.jpg'
        if valid(path):
            valid_idx.append(idx)
            
        if len(valid_idx) >= 15:
            break
    
    # split according to given rules
    if len(valid_idx) >= 15:
        train_idx = train_idx + valid_idx[:10]
        test_idx = test_idx + valid_idx[10:12]
        val_idx = val_idx + valid_idx[12:15]
    elif len(valid_idx) > 12:
        train_idx = train_idx + valid_idx[:10]
        test_idx = test_idx + valid_idx[10:12]
        val_idx = val_idx + valid_idx[12:]
    elif len(valid_idx) > 10:
        train_idx = train_idx + valid_idx[:10]
        test_idx = test_idx + valid_idx[10:]
    elif len(valid_idx) > 2:
        train_idx = train_idx + valid_idx[:-1]
        test_idx.append(valid_idx[-1])
    elif len(valid_idx) > 0:
        train_idx = train_idx + valid_idx

# Get image information
ids = train['id'].values
urls = train['url'].values
landmark_ids = train['landmark_id'].values

# Split training set
train_image_id = []
train_id = []
train_url = []
train_landmark_id = []

for idx in train_idx:
    from_path = './data/all/train_images/' + str(idx) + '.jpg'
    to_path = './data/triplet/train/' + str(idx) + '.jpg'
    img = io.imread(from_path)
    resized = np.array(resize(img, (256, 256, 3), mode='reflect') * 255, dtype=np.uint8)
    io.imsave(to_path, resized)
    train_image_id.append(idx)
    train_id.append(ids[idx])
    train_url.append(urls[idx])
    train_landmark_id.append(landmark_ids[idx])

# Save to disk   
train_df = pd.DataFrame({'image_id': train_image_id, 'id': train_id, 
                         'url': train_url, 'landmark_id': train_landmark_id})
train_df.to_csv('./data/triplet/train.csv', index=False, 
                columns=['image_id', 'id', 'url', 'landmark_id'])

# Split validation set
val_image_id = []
val_id = []
val_url = []
val_landmark_id = []

for idx in val_idx:
    from_path = './data/all/train_images/' + str(idx) + '.jpg'
    to_path = './data/triplet/validation/' + str(idx) + '.jpg'
    img = io.imread(from_path)
    resized = np.array(resize(img, (256, 256, 3), mode='reflect') * 255, dtype=np.uint8)
    io.imsave(to_path, resized)
    val_image_id.append(idx)
    val_id.append(ids[idx])
    val_url.append(urls[idx])
    val_landmark_id.append(landmark_ids[idx])

# Save to disk   
val_df = pd.DataFrame({'image_id': val_image_id, 'id': val_id, 
                       'url': val_url, 'landmark_id': val_landmark_id})
val_df.to_csv('./data/triplet/validation.csv', index=False, 
              columns=['image_id', 'id', 'url', 'landmark_id'])

# Split test set
test_image_id = []
test_id = []
test_url = []
test_landmark_id = []

for idx in test_idx:
    from_path = './data/all/train_images/' + str(idx) + '.jpg'
    to_path = './data/triplet/test/' + str(idx) + '.jpg'
    img = io.imread(from_path)
    resized = np.array(resize(img, (256, 256, 3), mode='reflect') * 255, dtype=np.uint8)
    io.imsave(to_path, resized)
    test_image_id.append(idx)
    test_id.append(ids[idx])
    test_url.append(urls[idx])
    test_landmark_id.append(landmark_ids[idx])

# Save to disk   
test_df = pd.DataFrame({'image_id': test_image_id, 'id': test_id, 
                        'url': test_url, 'landmark_id': test_landmark_id})
test_df.to_csv('./data/triplet/test.csv', index=False, 
               columns=['image_id', 'id', 'url', 'landmark_id'])

print('Train:\t\t', train_df.shape)
print('Validation:\t', val_df.shape)
print('Test:\t\t', test_df.shape)



