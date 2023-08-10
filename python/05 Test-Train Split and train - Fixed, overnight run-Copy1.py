import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from datetime import datetime
import pickle
import copy
import datetime

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

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

output_dir = "/home/ubuntu/data/TX_paired/"

geo_df = pickle.load( open( output_dir+"GeoDataFrame_fine_turked.pickled", "rb" ))
geo_df.rename(columns = {'post-storm_full':'post_resized','pre-storm_full':'pre_resized','post-storm_resized':'post_full','pre-storm_resized':'pre_full'}, inplace=True)
geo_df.set_index("tile_no")
geo_df.head(20)

INPUT_SIZE = 256

#helper functions:
def scale_bands(img, lower_pct = 1, upper_pct = 99):
    """Rescale the bands of a multichannel image for display"""
    img_scaled = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[2]):
        band = img[:, :, i]
        lower, upper = np.percentile(band, [lower_pct, upper_pct])
        band = (band - lower) / (upper - lower) * 255
        img_scaled[:, :, i] = np.clip(band, 0, 255).astype(np.uint8)
    return img_scaled

def resize(image, new_shape):
    #img_resized = np.zeros(new_shape+(img.shape[2],)).astype('float32')
    #for i in range(img.shape[2]):
    #    img_resized[:, :, i] = imresize(img[:, :, i], new_shape, interp='bicubic')
    #img_resized = cv2.resize(img,dsize=(new_shape,new_shape))
    
    r = new_shape / (1.0*image.shape[1])
    dim = (new_shape, int(image.shape[0] * r))
    
    if image.dtype == "int64": image = image.astype('float32')   #crashes if given 0 and 1 integers for some reason
    # perform the actual resizing of the image and show it
    img_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    #make training masks the proper dimensionality if only 2-D
    if len(image.shape) == 2: img_resized = np.expand_dims(img_resized,axis=2)

    return img_resized



#identify a training/testing set

geo_df.head(1)

#only use 'good' and verified files
good_geo_df = geo_df[geo_df.bad_image != True]
good_geo_df = good_geo_df[good_geo_df.verified == True]
#good_geo_df = good_geo_df[good_geo_df.tile_no <200]  #for testing with a smaller (faster) dataset
len(good_geo_df)



#note, these are the 256 resized images
image_files = sorted([output_dir+file+'.npy' for file in good_geo_df["post_resized"].values])  
before_image_files = sorted([output_dir+file+'.npy' for file in good_geo_df["pre_resized"].values])  
mask_files = sorted([output_dir+file+'.npy' for file in good_geo_df["DBScan_gauss"].values])


len(image_files),len(mask_files)
#min([len(image_files),len(mask_files)])



def TT_tile_no_split(good_geo_df,split=0.8):
    test_no = np.random.choice(good_geo_df.tile_no.values,int((1-split)*len(good_geo_df)))
    train_no = [num for num in good_geo_df.tile_no.values if num not in test_no]
    return train_no, test_no

train_no, test_no = TT_tile_no_split(good_geo_df,split=0.8)

len(train_no),len(test_no)

test_no

#write test and train tile_no lists to disk
f = open('/home/ubuntu/Notebooks/test_train_filelists/Unn_train_tile_numbers_'+str(datetime.datetime.now())+'.txt', 'w')
for item in train_no:f.write("%s\n" % item)
f.close()

f = open('/home/ubuntu/Notebooks/test_train_filelists/Unn_test_tile_numbers_'+str(datetime.datetime.now())+'.txt', 'w')
for item in test_no:f.write("%s\n" % item)
f.close()



output_dir+good_geo_df.post_resized[2]+'.npy'

def gen_train(good_geo_df,train_no):
    indexes = copy.deepcopy(train_no)
    while True:
        np.random.shuffle(indexes)

        for index in indexes:
            Xpost = np.load(output_dir+good_geo_df.post_resized[index]+'.npy')
            Xpre = np.load(output_dir+good_geo_df.pre_resized[index]+'.npy')
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
              
            Y = np.load(output_dir+good_geo_df.DBScan_gauss[index]+'.npy')
            Y = Y.astype('float32') #/ 255.
            #add extra first dimension for tensorflow compatability
            X = np.expand_dims(X,axis=0)
            Y = np.expand_dims(Y,axis=0)
            Y = np.expand_dims(Y,axis=3)
            yield (X,Y)

def gen_test(good_geo_df,test_no):
    indexes = copy.deepcopy(test_no)
    while True:
        np.random.shuffle(indexes)

        for index in indexes:
            Xpost = np.load(output_dir+good_geo_df.post_resized[index]+'.npy')
            Xpre = np.load(output_dir+good_geo_df.pre_resized[index]+'.npy')
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
              
            Y = np.load(output_dir+good_geo_df.DBScan_gauss[index]+'.npy')
            Y = Y.astype('float32') #/ 255.
            #add extra first dimension for tensorflow compatability
            X = np.expand_dims(X,axis=0)
            Y = np.expand_dims(Y,axis=0)
            Y = np.expand_dims(Y,axis=3)
            yield (X,Y)



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

def get_unet(lr=0.00001):
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



print("training/testing set size:",len(train_no),len(test_no))

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
# https://keras.io/callbacks/#reducelronplateau

# This sets the number of training epochs (you should do a lot more than 5)
NUM_EPOCHS = 500

# Define callback to save model checkpoints
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
model_checkpoint = ModelCheckpoint(os.path.join('checkpoints', 'weights.{epoch:02d}-{val_loss:.5f}.hdf5'), monitor='loss', save_best_only=True)

# Define callback to reduce learning rate when learning stagnates
# This won't actually kick in with only 5 training epochs, but I'm assuming you'll train for hundreds of epochs when you get serious about training this NN.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, epsilon=0.002, cooldown=2)

# # Define rate scheduler callack (this is an alternative to ReduceLROnPlateau. There is no reason to use both.)
# schedule = lambda epoch_i: 0.01*np.power(0.97, i)
# schedule_lr = LearningRateScheduler(schedule)

# TensorBoard visuluaziations... this stuff is so freaking cool
tensorboard = TensorBoard(log_dir='/tmp/tboard_logs2', histogram_freq=0, write_graph=True, write_images=True)

# Train the model
model = get_unet(1e-4)
#model.fit(X_train, Y_train, batch_size=32, epochs=NUM_EPOCHS, verbose=1, shuffle=True, callbacks=[model_checkpoint, reduce_lr, tensorboard], validation_data=(X_test, Y_test))
hist = model.fit_generator(gen_train(good_geo_df,train_no), steps_per_epoch=len(train_no), epochs=NUM_EPOCHS, verbose=1, shuffle=True,                     callbacks=[model_checkpoint, reduce_lr, tensorboard],                     validation_data=gen_train(good_geo_df,test_no), validation_steps = len(test_no),                    )#workers=6,use_multiprocessing=True)



hist.history.keys()

val_dice = hist.history['val_dice_coef']
val_loss = hist.history['val_loss']

fig = plt.figure(1)

# plot dice
plt.subplot(211)
plt.plot(val_dice)
plt.title('Validation dice coefficient and loss')
plt.ylabel('Dice coef value')

# plot loss
plt.subplot(212)
plt.plot(val_loss, 'r')
plt.xlabel('Epoch')
plt.ylabel('Loss value')

plt.show()















