import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import Counter
from datetime import datetime

import geopandas as gpd
import shapely.geometry
import rasterio
import json
import geopandas as gpd
import geopandas_osm.osm
from descartes import PolygonPatch
import h5py 
from scipy.misc import imresize
import shapely.geometry

import tensorflow as tf
import cv2
import pickle
import copy

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")



#footprint = "HOLDOUT/2131133.tif"
#footprint = "HOLDOUT/2131131.tif"
#footprint = "3020022.tif"

overlap_footprints = ['3020333.tif',
 '3020133.tif',
 '3020201.tif',
 '3020210.tif',
 '3020330.tif',
 '3020223.tif',
 '3002233.tif',
 '3020103.tif',
 '3020323.tif',
 '3020030.tif',
 '3020010.tif',
 '2113333.tif',
 '3020231.tif',
 '3002303.tif',
 '3020120.tif',
 '3020313.tif',
 '3020102.tif',
 '3002232.tif',
 '3020320.tif',
 '3020303.tif',
 '3020111.tif',
 '3020101.tif',
 '3020121.tif',
 '3020013.tif',
 '3020123.tif',
 '3002222.tif',
 '3020301.tif',
 '3002321.tif',
 '3020302.tif',
 '3020220.tif',
 '3020312.tif',
 '2131311.tif',
 '3002331.tif',
 '3020331.tif',
 '3020112.tif',
 '3002213.tif',
 '3020233.tif',
 '2131331.tif',
 '3002333.tif',
 '3020122.tif',
 '3020032.tif',
 '3020012.tif',
 '3020300.tif',
 '3020230.tif',
 '3020011.tif',
 '3002230.tif',
 '3020200.tif',
 '3020202.tif',
 '2131113.tif',
 '3020113.tif',
 '3020213.tif',
 '3002221.tif',
 '2131313.tif',
 '3020100.tif',
 '3020031.tif',
 '3020001.tif',
 '3002323.tif',
 '3020023.tif',
 '3020021.tif',
 '3002223.tif',
 '3020232.tif',
 '3020222.tif',
 '3002312.tif',
 '3020003.tif',
 '3020332.tif',
 '3020110.tif',
 '3020321.tif']

#choose a random footprint
footprint = str(np.random.choice(overlap_footprints,1)[0])
print(footprint)



#load a hold-out footprint:
src_post = rasterio.open('/home/ubuntu/data/TX_post/'+footprint)
big_img_post = src_post.read([1, 2, 3]).transpose([1,2,0])
src_pre = rasterio.open('/home/ubuntu/data/TX_pre/'+footprint)
big_img_pre = src_pre.read([1, 2, 3]).transpose([1,2,0])

width  = src_post.width
height = src_post.height
if src_post.height != src_pre.height: print("warning, different image heights")

#resize them by factor of n and plot (the raw files are 20K x 20K and take too long to display)
n = 20 #factor to reduce by
resize_width = big_img_post.shape[0]/n
r = resize_width / (1.0*big_img_post.shape[1])
dim = (resize_width, int(big_img_post.shape[0] * r))
big_img_post_resized = cv2.resize(big_img_post, dim, interpolation = cv2.INTER_AREA)
big_img_pre_resized  = cv2.resize(big_img_pre, dim, interpolation = cv2.INTER_AREA)
big_img_post_resized.shape, big_img_post_resized.dtype

plt.figure(figsize=(20,20))
plt.imshow(big_img_post_resized)
plt.show()

plt.figure(figsize=(20,20))
plt.imshow(big_img_pre_resized)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))
ax1.set_title("Pre-Flood DigitalGlobe Footprint")
ax2.set_title("Post-Flood DigitalGlobe Footprint")
ax2.axis('on')
ax1.imshow(big_img_pre)
ax2.imshow(big_img_post)



end_tilesize = INPUT_SIZE= 256

tiles_wide = width//(2*256)
tiles_high = height//(2*256)
print(tiles_wide,tiles_high)

#set up numpy arrays to store all the tiles

mosaic_raw_post = np.zeros((tiles_wide,tiles_high,2*end_tilesize,2*end_tilesize,3),dtype='uint8')
mosaic_raw_pre  = np.zeros((tiles_wide,tiles_high,2*end_tilesize,2*end_tilesize,3),dtype='uint8')
mosaic_raw_post.shape

tile_size = 2*end_tilesize
for i in range(0, width - tile_size, tile_size):  #note this i and j go 0, 512, 1024, etc.
    for j in range(0, height - tile_size, tile_size):
        window = ((j, j + tile_size), (i, i + tile_size))
        # Load the tile
        img_post = src_post.read(window=window).transpose([1,2,0])
        mosaic_raw_post[i/tile_size,j/tile_size] = img_post
        img_pre  =  src_pre.read(window=window).transpose([1,2,0])
        mosaic_raw_pre[i/tile_size,j/tile_size]  = img_pre

#this numpy arrage is huge!!! but my server has enough ram
sys.getsizeof(mosaic_raw_post)/1e9

plt.figure(figsize=(4,4))
plt.imshow(mosaic_raw_post[30,30]);

for i in range(mosaic_raw_post.shape[0]):
    for j in range(mosaic_raw_post.shape[1]):
        if mosaic_raw_post[i,j].sum(): print(i,j)

#set up numpy arrays to store all the tiles

mosaic_resize_post = np.zeros((tiles_wide,tiles_high,end_tilesize,end_tilesize,3),dtype='uint8')
mosaic_resize_pre  = np.zeros((tiles_wide,tiles_high,end_tilesize,end_tilesize,3),dtype='uint8')
mosaic_resize_post.shape

for i in range(mosaic_resize_post.shape[0]):
    for j in range(mosaic_resize_post.shape[1]):
        mosaic_resize_post[i,j] = cv2.resize(mosaic_raw_post[i,j], (end_tilesize,end_tilesize), interpolation = cv2.INTER_AREA)
        mosaic_resize_pre[i,j]  = cv2.resize(mosaic_raw_pre[i,j], (end_tilesize,end_tilesize), interpolation = cv2.INTER_AREA) 

plt.figure(figsize=(4,4))
plt.imshow(mosaic_resize_post[24, 30]);
print(sys.getsizeof(mosaic_resize_post)/1e9)
mosaic_resize_post[24,30].dtype







# Source: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py

import keras
from keras import backend as K
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, Nadam, Adamax, RMSprop

# Set network size params
N_CLASSES = 1
N_CHANNEL = 12

#dropout rate
out_per = 0.30

# Define metrics
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Just put a negative sign in front of an accuracy metric to turn it into a loss to be minimized
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def jacc_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def jacc_coef_loss(y_true, y_pred):
    return -jacc_coef(y_true, y_pred)

def jacc_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def get_unet(lr=1e-5):
    inputs = Input((INPUT_SIZE, INPUT_SIZE, N_CHANNEL))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Dropout(out_per)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Dropout(out_per)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Dropout(out_per)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Dropout(out_per)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Dropout(out_per)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = Dropout(out_per)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    conv7 = Dropout(out_per)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    conv8 = Dropout(out_per)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv9 = Dropout(out_per)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(axis = 1)(conv9)
    
    conv10 = Conv2D(N_CLASSES, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    # model.compile(optimizer=Adam(lr=lr), loss=jacc_coef_loss, metrics=[jacc_coef_int])
    # model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=[jacc_coef_int])
    # model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef, jacc_coef_int])

    return model

model = get_unet(1e-4)

get_ipython().system('ls checkpoints/')

model.load_weights('checkpoints/weights.16--0.69321.hdf5')



#numpy array for all the predictions
mosaic_prediction = np.zeros((tiles_wide,tiles_high,end_tilesize,end_tilesize)) #note, no color dimension

#convert an image to the correct format:

def ready_image(Xpost,Xpre):
    Xpre = Xpre.astype('float32')
    Xpost = Xpost.astype('float32')
    pre_mean = 92.36813   #taken from a central common image
    post_mean = 92.21524   #much closer than expected... are these representative?

    Xdiff = Xpost/post_mean - Xpre/pre_mean

    Xpost = (Xpost-post_mean)/post_mean  #divide by their respective means (per footprint would be even better)
    Xpre =  (Xpre-pre_mean)/pre_mean

    R,G,B = Xpost[:,:,0],Xpost[:,:,1],Xpost[:,:,2]
    Xratios_post = np.stack([R/G-1,R/B-1,G/B-1,R/(G+B)-0.5,G/(R+B)-0.5,B/(R+G)-0.5],axis=2)

    R,G,B = Xpre[:,:,0],Xpre[:,:,1],Xpre[:,:,2]
    Xratios_pre = np.stack([R/G-1,R/B-1,G/B-1,R/(G+B)-0.5,G/(R+B)-0.5,B/(R+G)-0.5],axis=2)

    #X = np.concatenate([Xpost,Xdiff,Xpre,Xratios_post,Xratios_pre],axis=2)
    X = np.concatenate([Xpost-1,Xdiff-1,Xratios_post],axis=2)
    
    return X

for i in range(mosaic_resize_post.shape[0]):
    for j in range(mosaic_resize_post.shape[1]):
        X = ready_image(mosaic_resize_post[i,j],mosaic_resize_pre[i,j])
        mosaic_prediction[i,j] = model.predict(X[None, ...])[0, ...,0]

#re-arrange the hypercube into a mosaic image
M = end_tilesize
mosaic_prediction_whole = np.zeros((width/2,height/2))
for i in range(mosaic_resize_post.shape[0]):
    for j in range(mosaic_resize_post.shape[1]):
        mosaic_prediction_whole[M*j : M*(j+1) , M*i : M*(i+1)] = mosaic_prediction[i,j] #note reversed i,j wierd huh

#resize them by factor of 2 to match prediction masks
n = 2 #factor to reduce by
resize_width = big_img_post.shape[0]/n
r = resize_width / (1.0*big_img_post.shape[1])
dim = (resize_width, int(big_img_post.shape[0] * r))
big_img_post_resized = cv2.resize(big_img_post, dim, interpolation = cv2.INTER_AREA)
big_img_pre_resized  = cv2.resize(big_img_pre, dim, interpolation = cv2.INTER_AREA)
big_img_post_resized.shape, big_img_post_resized.dtype

plt.figure(figsize=(16,16))
plt.imshow(big_img_post_resized)
plt.imshow(255*mosaic_prediction_whole,alpha=0.3)
plt.savefig('2131133_overlay.png');



plt.figure(figsize=(16,16))
plt.imshow(mosaic_prediction_whole)
plt.savefig('2131133_probmap.png')

plt.figure(figsize=(16,16))
plt.imshow(big_img_post_resized)
plt.savefig('2131133_post_flood_image.png')

plt.figure(figsize=(16,16))
plt.imshow(big_img_pre_resized)
plt.savefig('2131133_pre_flood_image.png')

plt.figure(figsize=(18,18))
plt.imshow(big_img_post_resized[2500:4500,0000:2000])
plt.imshow(255*(mosaic_prediction_whole[2500:4500,0000:2000]>0.5),alpha=0.33);



