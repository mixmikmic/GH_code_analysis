def create_heatmap(image_loc, model, height, downsample):
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    image = np.asarray(Image.open(image_loc))
    image_shape = image.shape
    image = image/255.0 # During training the images were normalized

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

import os
import pandas
import numpy as np
import glob
from scipy.ndimage import rotate
from PIL import Image
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')

import keras
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import metrics
from keras import layers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Reshape, Input, concatenate, Conv2DTranspose
from keras.layers.core import Activation, Dense, Lambda
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

def unet_standard(learning_rate=.0001):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10_dist = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    conv10_cross_entropy = Conv2D(3, (1, 1), activation='softmax')(conv9)
    output = concatenate([conv10_dist, conv10_cross_entropy])

    model = Model(img_input, output)
    model.compile(optimizer=Adam(lr=learning_rate), loss=distance_loss, metrics=[distance_loss])
    return model

def conv_block(x,
              filters,
              num_row,
              num_col,
              dropout, 
              padding='same',
              strides=(1, 1),
              activation='relu'):
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    return x

def unet_paper(learning_rate=.0001):
    input_shape = (None, None, 3)
    img_input = Input(shape=input_shape)

    conv1 = conv_block(img_input, 32, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 64, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 128, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 128, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    up5 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv5 = conv_block(up5, 128, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5), conv2], axis=3)
    conv6 = conv_block(up6, 64, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')

    up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv1], axis=3)
    conv7 = conv_block(up7, 32, 3, 3, dropout = .1, padding='same', strides=(1, 1), activation='relu')
    
    conv8_dist = Conv2D(1, (1, 1), activation='sigmoid')(conv7)
    conv8_cross_entropy = Conv2D(3, (1, 1), activation='softmax')(conv7)
    output = concatenate([conv8_dist, conv8_cross_entropy])
    
    model = Model(img_input, output)
    model.compile(optimizer=Adam(lr=learning_rate), loss=distance_loss, metrics=[distance_loss])
    return model


def distance_loss(y_true, y_pred):
    weight = .5 # how mush does the distance matter compared to the cross entropy (fast ai used .001 for 4 more uncertain ones)
    distance_loss = K.binary_crossentropy(y_pred[:, :, :, 0], y_true[:, :, :, 0])    
    cross_entropy = K.categorical_crossentropy(y_true[:, :, :, 1:], y_pred[:, :, :, 1:])    

    return(distance_loss*weight+(1-weight)*cross_entropy)

import keras
from keras.models import load_model, Model

model_loc = '/home/rbbidart/cancer_hist_out/unet_dist/unet_paper_custom_aug/unet_paper_0.005_custom_aug_.34-96.92.hdf5'
image_loc = '/home/rbbidart/project/rbbidart/cancer_hist/full_slides2/test/0/135_Region_3_crop.tif'
image = np.asarray(Image.open(image_loc))

model = unet_paper(learning_rate=.0001)
model.load_weights(model_loc)
image_pad = np.zeros((600, 1000, 3))
image_pad[:image.shape[0],:image.shape[1] , :] = image
image_pad = np.expand_dims(image_pad, axis=0)


heat = model.predict(image_pad, batch_size=1, verbose=0)
heat=np.squeeze(heat)
heat = heat[:image.shape[0],:image.shape[1] , :]
print(heat.shape)

f = plt.figure(figsize=(10,10))
plt.imshow(image)
f = plt.figure(figsize=(10,10))
plt.imshow(heat[:, :, 0])
f = plt.figure(figsize=(10,10))
plt.imshow(heat[:, :, 1])
f = plt.figure(figsize=(10,10))
plt.imshow(heat[:, :, 2])
f = plt.figure(figsize=(10,10))
plt.imshow(heat[:, :, 3])





