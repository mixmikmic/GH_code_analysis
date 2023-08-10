import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys, os, time, gc
import requests, shutil
from skimage import io
import keras
from keras.applications import InceptionV3

get_ipython().run_line_magic('matplotlib', 'inline')

# Read train and test dataframes
df_train = pd.read_csv('./data/resized/train_resized.csv')
df_test = pd.read_csv('./data/resized/test_resized.csv')

print('Train:\t\t', df_train.shape)
print('Test:\t\t', df_test.shape)

print('Landmarks:\t', len(df_train['landmark_id'].unique()))

# Use pre-trained Inception V3 model from Keras
inception = InceptionV3(include_top=False, weights='imagenet', input_shape=(256, 256, 3), pooling='avg')

inception.summary()

# Define train_imgs and test_imgs
train_imgs = np.zeros(shape=(len(df_train), 2048), dtype=np.float32)
test_imgs = np.zeros(shape=(len(df_test), 2048), dtype=np.float32)

# Process training images
steps = 50000
for i in range(0, len(df_train), steps):
    tmp_imgs = []
    print('\nProcess: {:10d}'.format(i))
    
    start = i
    end = min(len(df_train), i + steps)
    for idx in range(start, end):
        if idx % 1000 == 0:
            print('=', end='')
        img = io.imread('./data/resized/train_resized/' + str(idx) + '.jpg')
        tmp_imgs.append(img)
        
    tmp_imgs = np.array(tmp_imgs, dtype=np.float32) / 255.0
    tmp_prediction = inception.predict(tmp_imgs)
    train_imgs[start: end, ] = tmp_prediction
    _ = gc.collect()

# Process test images
steps = 20000
for i in range(0, len(df_test), steps):
    tmp_imgs = []
    print('\nProcess: {:10d}'.format(i))
    
    start = i
    end = min(len(df_test), i + steps)
    for idx in range(start, end):
        if idx % 1000 == 0:
            print('=', end='')
        img = io.imread('./data/resized/test_resized/' + str(idx) + '.jpg')
        tmp_imgs.append(img)
        
    tmp_imgs = np.array(tmp_imgs, dtype=np.float32) / 255.0
    tmp_prediction = inception.predict(tmp_imgs)
    test_imgs[start: end, ] = tmp_prediction
    _ = gc.collect()

# Save to disk
np.save('./data/resized/train_features.npy', train_imgs)
np.save('./data/resized/test_features.npy', test_imgs)

